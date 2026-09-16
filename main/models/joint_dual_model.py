import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.core.mixins import HyperparametersMixin
from diffusers.optimization import get_cosine_schedule_with_warmup
from main.models.autoencoders.GAE import GAE, GAE_PL
from main.models.diffusion.spaced_diff import SpacedDiffusion
from main.util import *


class ModelWrapperV2(pl.LightningModule, HyperparametersMixin):
    def __init__(
        self,
        online_network=None,
        target_network=None,
        encoder=None,
        Gae_pl=None,
        config=None,
        optimizer_name="Adam",
        lr=1e-5,
        lr_scheduler='warmup',
        training_steps=60_000,
        weight_decay=0.01,
        betas=None,
        cfd_rate=0.0,
        warmup=0,
        loss="l2",
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
        lambda_eps=1.,
        lambda_x0=1.,
        lambda_adaptive_factor=1.,
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
        assert sample_method in ["ddpm", "ddim", "dpm_solver"]
        assert skip_strategy in ["uniform", "quad"]
        assert train_pattern in ['encoder+GAE+ddpm', 'encoder+ddpm', 'ddpm']
        ignore_list = [
            # 'online_network',
            # 'target_network',
            # 'encoder',
        ]
        ignore_list = ignore_list.append('Gae_pl') if Gae_pl is not None else ignore_list
        # self.save_hyperparameters(ignore=ignore_list)
        # self.supported_data = ['fonts', 'ilab_128', "fonts_group", "ilab_group"]
        self.config = config
        self.use_z = use_z
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer_name = optimizer_name
        self.pred_mode = self.online_network.decoder.pred_mode
        self.use_adaptive = self.online_network.decoder.use_adaptive
        self.use_ada_model = self.online_network.decoder.use_ada_model
        self.is_group_training = is_group_training
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
        self.p_loss = LPIPSwithPixel(
            lpips_loss_weight=2,
            pixel_loss_weight=1,
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.training_steps = training_steps
        self.lr = lr
        self.betas = betas
        self.lr_schedulers_name = lr_scheduler
        self.weight_decay = weight_decay
        self.grad_clip_val = grad_clip_val
        self.warmup = warmup
        self.gamma = loss_weight
        self.lambda_eps = lambda_eps
        self.lambda_x0 = lambda_x0
        self.lambda_adaptive_factor = lambda_adaptive_factor
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

        # Predict noise, x_0 and mixture coefficient
        pred, x_t = self.online_network(
            x, eps, t, low_res=None, z=z.squeeze() if self.use_z else None,
        )

        # Compute loss
        # loss_eps, loss_x0, loss_adaptive_factor = 0., 0., 0.
        if self.use_adaptive:
            eps_pred, x0_pred, adaptive_factor = torch.split(pred, [3, 3, 1], dim=1)
            if self.pred_mode == 'adaptive_form1':
                real_xt_minus1 = self.online_network.get_posterior_mean_from_noise(
                    x_t,
                    t,
                    _eps_score=eps,
                    cond=None,
                    z=None,
                )
                pred_xt_minus1_from_noise = self.online_network.get_posterior_mean_from_noise(
                    x_t,
                    t,
                    _eps_score=eps_pred,
                    cond=None,
                    z=None,
                )
                pred_xt_minus1_from_x0 = self.online_network.get_posterior_mean_from_x0(
                    x_t,
                    t,
                    _x0_score=x0_pred,
                    clip_denoised=True,
                    cond=None,
                    z=None,
                )

                pred_xt_minus1 = adaptive_factor * pred_xt_minus1_from_noise.detach() + \
                                 (1 - adaptive_factor) * pred_xt_minus1_from_x0.detach()

                loss_eps = self.criterion(eps, eps_pred)
                loss_x0 = self.p_loss(x, x0_pred)
                loss_adaptive_factor = self.criterion(real_xt_minus1, pred_xt_minus1)

            elif self.pred_mode == 'adaptive_form2':
                eps_from_x0_pred = self.online_network.get_posterior_eps_from_xt_x0(
                    x_t=x_t,
                    t_=t,
                    x_0=x0_pred,
                    clip_denoised=False,
                    cond=None,
                    z=None,
                )
                pred_eps = (1 - adaptive_factor) * eps_from_x0_pred.detach() + adaptive_factor * eps_pred.detach()
                loss_eps = self.criterion(eps, eps_pred)
                loss_x0 = self.p_loss.forward(x, x0_pred)
                loss_adaptive_factor = self.criterion(eps, pred_eps)
            else:
                raise NotImplementedError()
        elif self.pred_mode == 'eps':
            loss_eps = self.criterion(eps, pred)
            loss_x0 = 0.
            loss_adaptive_factor = 0.
        elif self.pred_mode == 'x0':
            loss_x0 = self.criterion(x, pred)
            loss_eps = 0.
            loss_adaptive_factor = 0.
        else:
            raise ValueError(f'{self.pred_mode} is not supported')

        loss_ddpm = self.lambda_eps * loss_eps + self.lambda_x0 * loss_x0 + self.lambda_adaptive_factor * loss_adaptive_factor
        whole_loss = loss_ddpm + self.gamma * loss_GAE  # alpha should be very small when fine-tuning, otherwise numerical instability

        if self.gamma != 0:
            self.log('GAE_loss', loss_GAE, prog_bar=True)
        if self.use_adaptive:
            self.log("loss_eps", loss_eps, prog_bar=True)
            self.log("loss_x0", loss_x0, prog_bar=True)
            self.log("loss_adaptive_factor", loss_adaptive_factor, prog_bar=True)
        self.log('whole_loss', whole_loss, prog_bar=True)
        self.log('global_steps', self.global_step, prog_bar=True)
        return whole_loss

    def configure_optimizers(self):
        out = {}
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.online_network.decoder.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.online_network.decoder.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        else:
            raise NotImplementedError()
        out['optimizer'] = optimizer
        if self.lr_schedulers_name == 'cosine_warmup':
            # cosine lr_scheduler with warmup
            assert self.warmup > 0, f'warmup_step should be greater than 0'
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.warmup,
                num_training_steps=self.training_steps,
                last_epoch=-1,
            )
            out['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'step',
            }
        elif self.lr_schedulers_name == 'warmup':
            # const lr_scheduler with warmup
            lr_lambda = WarmupLR(self.warmup)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            out['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'step',
            }
        else:
            raise NotImplemented(f'{self.lr_schedulers_name} is not implemented')

        return out
