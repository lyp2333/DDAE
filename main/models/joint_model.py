import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.cuda import amp
from pytorch_lightning.core.mixins import HyperparametersMixin
from diffusers.optimization import get_cosine_schedule_with_warmup
from main.models.autoencoders.GAE import GAE, GAE_PL
from main.models.diffusion.spaced_diff import SpacedDiffusion
from main.models.diffusion import DDPM
from main.util import *


class ModelWrapper(pl.LightningModule, HyperparametersMixin):
    def __init__(
        self,
        online_network,
        target_network,
        encoder,
        Gae_pl=None,
        config=None,
        lr=1e-5,
        lr_scheduler='warmup',
        training_steps=60_000,
        weight_decay=0.01,
        cfd_rate=0.0,
        warmup=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddim",
        conditional=True,
        eval_mode="recons",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        use_z=False,
        ddpm_latents=None,
        loss_weight=0.,
        sample_storage_path=None,
        train_pattern='encoder+ddpm',
        latent_z_mean_std=None,
        is_group_training=False,
        group_size=None,
        batch_size=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]
        assert train_pattern in ['encoder+GAE+ddpm', 'encoder+ddpm', 'ddpm']
        ignore_list = [
            'online_network',
            'target_network',
            'encoder',
        ]
        ignore_list = ignore_list.append('Gae_pl') if Gae_pl is not None else ignore_list
        # self.save_hyperparameters(ignore=ignore_list)
        # self.supported_data = ['fonts', 'ilab_128', "fonts_group", "ilab_group"]
        self.config = config
        self.use_z = use_z
        self.online_network = online_network
        self.target_network = target_network
        self.pred_mode = self.online_network.decoder.pred_mode
        assert self.pred_mode in ["eps", "x0"]
        self.is_group_training = is_group_training
        self.training_steps = training_steps
        self.sample_storage_path = sample_storage_path
        self.cfd_rate = cfd_rate

        # config_joint
        # self.config = config
        self.train_pattern = train_pattern
        # Gae initialization
        if self.train_pattern == "encoder+GAE+ddpm":
            self.encoder = encoder
            assert Gae_pl is not None
            assert latent_z_mean_std is not None
            self.Gae_pl = Gae_pl
            self.z_mean = latent_z_mean_std['mean']
            self.z_std = latent_z_mean_std['std']
        # diffusion autoencoder
        elif self.train_pattern == "encoder+ddpm":
            self.encoder = encoder
        else:
            raise NotImplementedError()

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.batch_size = batch_size
        self.group_size = group_size
        self.lr = lr
        self.lr_schedulers_name = lr_scheduler
        self.weight_decay = weight_decay
        self.grad_clip_val = grad_clip_val
        self.warmup = warmup
        self.gamma = loss_weight
        self.sample_storage_path = sample_storage_path
        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # setting for space_diffusion
        self.spaced_diffusion = None
        # DDPM and DDPM_form2 as base_samplers which have trained, style of DDPM-sample use it
        self.base_sampler = self.target_network if self.sample_from == 'target' else self.online_network

        # SpaceDif and SpaceDif_form2 which have base_samples's betas improve speed of sample process,
        # style of DDIM use it

        # # Spaced Diffusion (for spaced re-sampling)
        if self.resample_strategy == 'spaced':
            spaced_sampler = SpacedDiffusion
            # self.pred_steps represents not only final sample times for ddim, but medium sample times in ddpm
            num_steps = self.pred_steps if self.pred_steps is not None else self.online_network.T
            indices = space_timesteps(self.base_sampler.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_sampler(self.base_sampler, indices)
        else:
            # for truncated resampling
            if self.sample_method == "ddim":
                raise ValueError("DDIM is only supported for spaced sampling")

    # @property
    # def batch_size(self):
    #     r'''
    #
    #     :return: batch_size for each dataset
    #     '''
    #     batch_size = self.config.training.batch_size
    #     if self.is_group_training:
    #         if self.data_type == 'fonts':
    #             # not for distributed training
    #             return batch_size * 6
    #         elif self.data_type == 'ilab_20M':
    #             return batch_size * 4
    #         else:
    #             raise NotImplemented("Not implemented")
    #     else:
    #         return batch_size

    @property
    def num_samples(self):
        return self.global_step * self.batch_size

    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        r"""
        get samples from x_t and cond(if exists) or z(if exists)
        :param x:
        :param cond:
        :param z:
        :param n_steps:
        :param ddpm_latents:
        :param checkpoints:
        :return:
        """

        if self.resample_strategy == 'spaced':
            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            else:
                # ddpm sample
                return self.spaced_diffusion(
                    x,
                    cond=cond,
                    z=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                    ddpm_latents=ddpm_latents,
                )
        # for truncated resampling
        else:
            if self.sample_method == "ddim":
                raise ValueError("DDIM is only supported for spaced sampling")
            return self.base_sampler.sample(
                x,
                cond=cond,
                z=z,
                n_steps=n_steps,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

    def training_step(self, batch, batch_idx):
        x = batch.flatten(0, 1) if self.is_group_training else batch
        # init loss_GAE value
        loss_GAE = 0.
        # loss_ddpm
        cond = None
        z = None
        if self.conditional:
            # three patterns, encoder+decoder+ddpm, encoder+ddpm, ddpm
            if self.train_pattern == 'encoder+GAE+ddpm':
                # To use training_step in GAE_PL, class ModelWrapper must have the same attribute named 'Gae'
                z = self.encoder(x)
                z = (z - self.z_mean) / self.z_std
                terms = self.Gae_pl.training_step(z.reshape(self.batch_size, self.group_size, -1), -1)
                loss_GAE, z_recon, _ = terms['loss'], terms['z_recon'], terms['z_dis']
                z = z_recon.flatten(0, 1)
                z = z * self.z_std + self.z_mean
            elif self.train_pattern == 'encoder+ddpm':
                z = self.encoder(x)
            else:
                raise NotImplemented(f'{self.train_pattern} is not implemented')
        else:
            raise ValueError('conditional should be true')
        # Sample timepoints randomly
        t = torch.randint(
            0,
            self.online_network.T,
            size=(x.size(0),),
            device=self.device,
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise or x_0
        pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.use_z else None,
        )

        # Compute loss
        target = eps if self.pred_mode == 'eps' else x
        loss_ddpm = self.criterion(target, pred)
        whole_loss = loss_ddpm + self.gamma * loss_GAE

        self.log('GAE_loss', loss_GAE, prog_bar=True)
        self.log('whole_loss', whole_loss, prog_bar=True)
        return whole_loss

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.grad_clip_val > 0:
            params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params,
                                           self.grad_clip_val,
                                           )

    def configure_optimizers(self):
        out = {}
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        out['optimizer'] = optimizer
        if self.lr_schedulers_name == 'cosine_warmup':
            assert self.warmup > 0, f'warmup_step should be greater than 0'
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.warmup,
                num_training_steps=self.training_steps,
            )
            out['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'step',
            }
        elif self.lr_schedulers_name == 'warmup':
            lr_lambda = WarmupLR(self.warmup)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            out['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'step',
            }
        else:
            raise NotImplemented(f'{self.lr_schedulers_name} is not implemented')

        return out
