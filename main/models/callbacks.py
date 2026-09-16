import os
from typing import Sequence, Union, Any
from PIL import Image
import torch
from pytorch_lightning import Callback, LightningModule, Trainer, loggers
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning import loggers as pl_loggers
from torch import Tensor
from torch.nn import Module
from torchvision.utils import make_grid, save_image
from ..util import save_as_images, save_as_np, get_named_beta_schedule
from ..models.diffusion.dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper
from ..models.diffusion.unet_openai import DecoderWrapper


class EMAWeightUpdate(Callback):
    """EMA weight update
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, tau: float = 0.9999):
        """
        Args:
            tau: EMA decay rate
        """
        super().__init__()
        self.tau = tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_network.decoder
        target_net = pl_module.target_network.decoder

        # update weights
        self.update_weights(online_net, target_net)

    def update_weights(
        self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]
    ) -> None:
        # apply EMA weight update
        with torch.no_grad():
            for targ, src in zip(target_net.parameters(), online_net.parameters()):
                targ.mul_(self.tau).add_(src, alpha=1 - self.tau)


class ImageWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        compare=False,
        n_steps=None,
        eval_mode="sample",
        conditional=True,
        sample_prefix="",
        save_vae=False,
        save_mode="image",
        is_norm=True,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons"]
        self.output_dir = output_dir
        self.compare = compare
        self.n_steps = 1000 if n_steps is None else n_steps
        self.eval_mode = eval_mode
        self.conditional = conditional
        self.sample_prefix = sample_prefix
        self.save_vae = save_vae
        self.is_norm = is_norm
        self.save_fn = save_as_images if save_mode == "image" else save_as_np

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        rank = pl_module.global_rank
        if self.conditional:
            ddpm_samples_dict, vae_samples = prediction

            if self.save_vae:
                vae_samples = vae_samples.cpu()
                vae_save_path = os.path.join(self.output_dir, "vae")
                os.makedirs(vae_save_path, exist_ok=True)
                self.save_fn(
                    vae_samples,
                    file_name=os.path.join(
                        vae_save_path,
                        f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}",
                    ),
                    denorm=self.is_norm,
                )
        else:
            ddpm_samples_dict = prediction

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        for k, ddpm_samples in ddpm_samples_dict.items():
            ddpm_samples = ddpm_samples.cpu()

            # Setup dirs
            base_save_path = os.path.join(self.output_dir, k)
            img_save_path = os.path.join(base_save_path, "images")
            os.makedirs(img_save_path, exist_ok=True)

            # Save
            self.save_fn(
                ddpm_samples,
                file_name=os.path.join(
                    img_save_path, f"output_{self.sample_prefix}_{rank}_{batch_idx}"
                ),
                denorm=self.is_norm,
            )


class LogSample(Callback):
    """
        callback for log samples in training
    """

    def __init__(self, num_samples, sample_every_step, storage_path=None):
        self.num_samples = num_samples
        self.sample_every_step = sample_every_step
        self.storage_path = storage_path

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.config.training.sample_every_step > 0 and self._is_time(pl_module.global_step, pl_module.config.training.sample_every_step):
            print('\nGenerating samples, please wait\n', flush=True)
            group_size = pl_module.config.data.group_size
            is_group_dataset = pl_module.config.data.is_group_dataset
            nrows = group_size if group_size > 0 and is_group_dataset else pl_module.config.training.num_samples // 2
            norm = pl_module.config.training.samples_norm
            sample_from = pl_module.sample_from
            sample_path = os.path.join(pl_module.sample_storage_path, f'{pl_module.config.training.training_name}--step_{pl_module.global_step}')
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            logger = pl_module.logger
            global_step = pl_module.global_step
            d_name = pl_module.config.data.name
            use_adaptive = pl_module.use_adaptive
            if use_adaptive:
                if 'z' not in d_name:
                    train_pattern = pl_module.train_pattern
                    if train_pattern == 'encoder+GAE+ddpm':
                        # todo change
                        diffusion_sample, diffusion_randn_sample, original_imgs = self._get_adaptive_model_samples(pl_module,
                                                                                                                   batch,
                                                                                                                   self.num_samples,
                                                                                                                   )
                        self._save_to_local_and_tb(logger, global_step, original_imgs, 'original', os.path.join(sample_path, 'original.png'), sample_from=sample_from, nrow=nrows, norm=norm),
                        self._save_to_local_and_tb(logger, global_step, diffusion_sample, 'Diffusion_recon', os.path.join(sample_path, 'Diffusion_recon.png'), sample_from=sample_from, nrow=nrows,
                                                   norm=norm)
                        self._save_to_local_and_tb(logger, global_step, diffusion_randn_sample, 'random_sample_recon', os.path.join(sample_path, 'random_sample_recon.png'), sample_from=sample_from,
                                                   nrow=nrows, norm=norm)

                    elif train_pattern == 'encoder+ddpm':
                        # todo change

                        diffusion_sample_from_adaptive, diffusion_randn_sample, original_imgs, diffusion_sample_from_eps, diffusion_sample_from_x0 = self._get_adaptive_model_samples(pl_module,
                                                                                                                                                                                      batch,
                                                                                                                                                                                      self.num_samples,
                                                                                                                                                                                      )

                        self._save_to_local_and_tb(logger, global_step, original_imgs, 'original', os.path.join(sample_path, 'original.png'), sample_from=sample_from, nrow=nrows, norm=norm),
                        self._save_to_local_and_tb(logger, global_step, diffusion_sample_from_adaptive, 'Diffusion_recon_adaptive', os.path.join(sample_path, 'Diffusion_recon.png'),
                                                   sample_from=sample_from, nrow=nrows, norm=norm)
                        self._save_to_local_and_tb(logger, global_step, diffusion_sample_from_eps, 'Diffusion_recon_eps', os.path.join(sample_path, 'Diffusion_recon.png'),
                                                   sample_from=sample_from, nrow=nrows, norm=norm)
                        self._save_to_local_and_tb(logger, global_step, diffusion_sample_from_x0, 'Diffusion_recon_x0', os.path.join(sample_path, 'Diffusion_recon.png'),
                                                   sample_from=sample_from, nrow=nrows, norm=norm)
                        self._save_to_local_and_tb(logger, global_step, diffusion_randn_sample, 'random_sample_recon', os.path.join(sample_path, 'random_sample_recon.png'), sample_from=sample_from,
                                                   nrow=nrows, norm=norm)

                elif 'z' in d_name:
                    # todo change
                    ...

            else:
                if 'z' not in d_name:
                    train_pattern = pl_module.train_pattern
                    if train_pattern == 'encoder+GAE+ddpm':
                        diffusion_sample, diffusion_randn_sample, original_imgs = self._get_samples(pl_module,
                                                                                                    batch,
                                                                                                    self.num_samples,
                                                                                                    )
                        self._save_to_local_and_tb(logger, global_step, original_imgs, 'original', os.path.join(sample_path, 'original.png'), sample_from=sample_from, nrow=nrows, norm=norm),
                        self._save_to_local_and_tb(logger, global_step, diffusion_sample, 'Diffusion_recon', os.path.join(sample_path, 'Diffusion_recon.png'), sample_from=sample_from, nrow=nrows,
                                                   norm=norm)
                        self._save_to_local_and_tb(logger, global_step, diffusion_randn_sample, 'random_sample_recon', os.path.join(sample_path, 'random_sample_recon.png'), sample_from=sample_from,
                                                   nrow=nrows, norm=norm)

                    elif train_pattern == 'encoder+ddpm':
                        diffusion_sample, diffusion_randn_sample, original_imgs = self._get_samples(pl_module,
                                                                                                    batch,
                                                                                                    self.num_samples,
                                                                                                    )

                        self._save_to_local_and_tb(logger, global_step, original_imgs, 'original', os.path.join(sample_path, 'original.png'), sample_from=sample_from, nrow=nrows, norm=norm),
                        self._save_to_local_and_tb(logger, global_step, diffusion_sample, 'Diffusion_recon', os.path.join(sample_path, 'Diffusion_recon.png'), sample_from=sample_from, nrow=nrows,
                                                   norm=norm)
                        self._save_to_local_and_tb(logger, global_step, diffusion_randn_sample, 'random_sample_recon', os.path.join(sample_path, 'random_sample_recon.png'), sample_from=sample_from,
                                                   nrow=nrows,
                                                   norm=norm)
                elif 'z' in d_name:
                    # save images to disk
                    sample_size = min(len(batch), self.num_samples)
                    batch_size, group_size, z_dim = batch[:sample_size].shape
                    z_original = batch[:sample_size].flatten(0, 1)
                    z_recon = outputs['z_recon'][:sample_size].flatten(0, 1)
                    dev = z_recon.device
                    if pl_module.config.data.norm:
                        z_mean, z_std = pl_module.z_mean.to(dev), pl_module.z_std.to(dev)
                        real_z_original = (z_original * z_std) + z_mean
                        real_z_recon = (z_recon * z_std) + z_mean
                    else:
                        real_z_original = z_original
                        real_z_recon = z_recon
                    sampler = pl_module.sampler
                    img_size = pl_module.config.data.image_size
                    noise = torch.randn(sample_size * group_size, 3, img_size, img_size).to(dev)
                    pred_step = len(sampler.timestep_map)
                    with torch.no_grad():
                        ddim_recon_original = sampler.ddim_sample(
                            x_t=noise,
                            cond=None,
                            z=real_z_original,
                            eta=0.,
                            guidance_weight=0.,
                        )[str(pred_step)]
                        ddim_recon_randn = sampler.ddim_sample(
                            x_t=noise,
                            cond=None,
                            z=real_z_recon,
                            eta=0.,
                            guidance_weight=0.,
                        )[str(pred_step)]
                    self._save_to_local_and_tb(logger,
                                               global_step,
                                               ddim_recon_randn,
                                               "random_sample_recon",
                                               os.path.join(sample_path,
                                                            'random_sample_recon.png'),
                                               sample_from=sample_from,
                                               nrow=nrows,
                                               norm=norm)
                    self._save_to_local_and_tb(logger,
                                               global_step,
                                               ddim_recon_original,
                                               "original",
                                               os.path.join(sample_path,
                                                            'original.png'),
                                               sample_from=sample_from,
                                               nrow=nrows,
                                               norm=norm)
                else:
                    raise NotImplementedError()

    @staticmethod
    def _is_time(global_step: int, interval: int):
        """
        judge whether to log samples, We should generate samples when functional returns True
        :param global_step:
        :param interval:
        :return: bool
        """
        return not (global_step % interval)

    @staticmethod
    @torch.no_grad()
    def _get_samples(model: LightningModule, batch: dict, num_samples: int):
        """
        put images to tensorboard and save to results dir
        :param model:
        :param path:
        :param batch:
        :param num_samples:
        :return: (sample by diffusion, sample by GAE, original images)
        """
        # Generate samples
        batch = batch.flatten(0, 1) if model.is_group_training else batch
        sample_size = min(len(batch), num_samples)
        img = batch[:sample_size]
        cond = None
        z = model.encoder(img)
        if model.train_pattern == 'encoder+ddpm':
            if model.resample_strategy == 'spaced':
                if model.sample_method == 'ddim':
                    # deterministic process
                    x_t = model.spaced_diffusion.ddim_reverse_sample_loop(
                        x=img,
                        cond=cond,
                        z=z,
                        eta=0.,
                    )['sample']

                    ddim_recon = model.spaced_diffusion.ddim_sample(
                        x_t=x_t,
                        cond=cond,
                        z=z,
                        eta=0.,
                        guidance_weight=0.,
                    )[str(model.pred_steps)]

                    # x_T sample from N~(0, I)
                    x_t = torch.randn(x_t.shape, device=x_t.device)
                    ddim_recon_randn = model.spaced_diffusion.ddim_sample(
                        x_t=x_t,
                        cond=cond,
                        z=z,
                        eta=0.,
                        guidance_weight=0.,
                    )[str(model.pred_steps)]
                    return ddim_recon, ddim_recon_randn, img
                elif model.sample_method == 'dpm_solver':
                    # init dpm-solver sampler
                    beta_scheduler = torch.from_numpy(get_named_beta_schedule('linear', 1000))
                    noise_scheduler = NoiseScheduleVP(
                        schedule='discrete',
                        betas=beta_scheduler,
                    )
                    model_fn = model_wrapper(
                        model=model.spaced_diffusion.decoder,
                        noise_schedule=noise_scheduler,
                        model_type='x_start',
                        guidance_type='classifier-free',
                        condition=z,
                        guidance_scale=1.,
                    )
                    dpm_solver = DPM_Solver(
                        model_fn=model_fn,
                        noise_schedule=noise_scheduler,
                        algorithm_type="dpmsolver++",
                    )
                    # get inversed noise and random noise for sampling
                    x_t_reversed = dpm_solver.inverse(
                        x=img,
                        order=3,
                        skip_type='time_uniform',
                        method='multistep',
                    )
                    x_t_random = torch.randn(img.shape, device=img.device)
                    # get samples from dpm-solver
                    # default value for t_start and t_end is exactly sampling
                    revered_recon = dpm_solver.sample(
                        x=x_t_reversed,
                        steps=20,
                        order=3,
                        skip_type='time_uniform',
                        method='multistep',
                        return_intermediate=False,
                    )
                    random_recon = dpm_solver.sample(
                        x=x_t_random,
                        steps=20,
                        order=2,
                        skip_type='time_uniform',
                        method='multistep',
                        return_intermediate=False,
                    )
                    return revered_recon, random_recon, img
                else:
                    raise NotImplemented('not implemented')
            else:
                raise NotImplemented('not implemented')

        elif model.train_pattern == "encoder+GAE+ddpm":
            z_mean, z_std = model.z_mean, model.z_std
            z = (z - z_mean) / z_std
            Gae_recon, z = model.Gae_pl(z)
            z = Gae_recon * z_std + z_mean

            # deterministic x_T
            x_t = model.spaced_diffusion.ddim_reverse_sample_loop(
                x=img,
                cond=cond,
                z=z,
                eta=0.,
            )['sample']
            ddim_recon = model.spaced_diffusion.ddim_sample(
                x_t=x_t,
                cond=cond,
                z=z,
                eta=0.,
                guidance_weight=0.,
            )[str(model.pred_steps)]

            # ---- "w/ implicit constraint" reconstruction (paper Table 1) ----
            x_t_con = model.spaced_diffusion.ddim_reverse_sample_loop(
                x=img,
                cond=cond,
                z=z,
                eta=0.,
                use_implicit_constraint=True,
            )['sample']
            ddim_recon_con = model.spaced_diffusion.ddim_sample(
                x_t=x_t_con,
                cond=cond,
                z=z,
                eta=0.,
                guidance_weight=0.,
                use_implicit_constraint=True,
            )[str(model.pred_steps)]

            # x_T sample from N~(0, I)
            x_t = torch.randn(x_t.shape, device=x_t.device)
            ddim_recon_randn = model.spaced_diffusion.ddim_sample(
                x_t=x_t,
                cond=cond,
                z=z,
                eta=0.,
                guidance_weight=0.,
            )[str(model.pred_steps)]
            return ddim_recon, ddim_recon_randn, img, ddim_recon_con
        else:
            raise NotImplemented(f'callback for {model.train_pattern} is not implemented')

    @staticmethod
    @torch.no_grad()
    def _get_adaptive_model_samples(model: LightningModule, batch: torch.Tensor, num_samples: int):
        # Generate samples
        batch = batch.flatten(0, 1) if model.is_group_training else batch
        sample_size = min(len(batch), num_samples)
        img = batch[:sample_size]
        cond = None

        z = model.encoder(img)
        if model.train_pattern == 'encoder+ddpm':
            if model.resample_strategy == 'spaced':
                if model.sample_method == 'ddim':
                    # deterministic process
                    x_t = model.spaced_diffusion.ddim_reverse_sample_loop(
                        x=img,
                        cond=cond,
                        z=z,
                        eta=0.,
                    )['sample']

                    ddim_recon_dict = model.spaced_diffusion.ddim_sample(
                        x_t=x_t,
                        cond=cond,
                        z=z,
                        clip_denoised="static",
                        eta=0.,
                        guidance_weight=0.,
                    )
                    ddim_recon = ddim_recon_dict[str(model.pred_steps)]
                    # with open("/home/lyp/Code/adaptive_factor.txt", 'a+') as f:
                    #     for i in range(model.pred_steps):
                    #         f.write("i + 1" + " " + str(ddim_recon_dict[f"adaptive_factor-{i+1}"])+'\n')

                    ddim_recon_from_eps = model.spaced_diffusion.ddim_sample(
                        x_t=x_t,
                        cond=cond,
                        z=z,
                        clip_denoised="static",
                        adaptive_factor=1.,
                        eta=0.,
                        guidance_weight=0.,
                    )[str(model.pred_steps)]

                    ddim_recon_from_x0 = model.spaced_diffusion.ddim_sample(
                        x_t=x_t,
                        cond=cond,
                        z=z,
                        clip_denoised="static",
                        adaptive_factor=0.,
                        eta=0.,
                        guidance_weight=0.,
                    )[str(model.pred_steps)]

                    # x_T sample from N~(0, I)
                    x_t = torch.randn(img.shape, device=img.device)
                    ddim_recon_randn = model.spaced_diffusion.ddim_sample(
                        x_t=x_t,
                        cond=cond,
                        z=z,
                        clip_denoised="static",
                        eta=0.,
                        guidance_weight=0.,
                    )[str(model.pred_steps)]

                    return ddim_recon, ddim_recon_randn, img, ddim_recon_from_eps, ddim_recon_from_x0
                elif model.sample_method == 'dpm_solver':
                    beta_scheduler = torch.from_numpy(get_named_beta_schedule('linear', 1000)).to(batch.device)
                    noise_scheduler = NoiseScheduleVP(
                        schedule='discrete',
                        betas=beta_scheduler,
                    )
                    # wrap model from adaptive prediction to eps prediction
                    model_wrapped = DecoderWrapper(
                        original_decoder=model.spaced_diffusion.decoder,
                        betas=beta_scheduler,
                        target_predict_type="eps",
                    )
                    model_fn = model_wrapper(
                        model_wrapped,
                        model_type="noise",
                        guidance_type='classifier-free',
                        condition=z,
                        guidance_scale=1.,
                    )
                    dpm_solver = DPM_Solver(
                        model_fn=model_fn,
                        noise_schedule=noise_scheduler,
                        algorithm_type="dpmsolver++",
                    )
                    # get inversed noise and random noise for sampling
                    x_t_reversed = dpm_solver.inverse(
                        x=img,
                        order=3,
                        skip_type='time_uniform',
                        method='multistep',
                    )
                    x_t_random = torch.randn(img.shape, device=img.device)
                    # get samples from dpm-solver
                    # default value for t_start and t_end is exactly sampling
                    revered_recon = dpm_solver.sample(
                        x=x_t_reversed,
                        steps=20,
                        order=3,
                        skip_type='time_uniform',
                        method='multistep',
                        return_intermediate=False,
                    )
                    random_recon = dpm_solver.sample(
                        x=x_t_random,
                        steps=20,
                        order=2,
                        skip_type='time_uniform',
                        method='multistep',
                        return_intermediate=False,
                    )
                    return revered_recon, random_recon, img
                else:
                    raise NotImplemented('not implemented')
            else:
                # todo add ddpm sampling
                raise NotImplemented('not implemented')

        elif model.train_pattern == "encoder+GAE+ddpm":
            z_mean, z_std = model.z_mean, model.z_std
            z = (z - z_mean) / z_std
            Gae_recon, z = model.Gae_pl(z)
            z = Gae_recon * z_std + z_mean

            # deterministic x_T
            x_t = model.spaced_diffusion.ddim_reverse_sample_loop(
                x=img,
                cond=cond,
                z=z,
                eta=0.,
            )['sample']
            ddim_recon = model.spaced_diffusion.ddim_sample(
                x_t=x_t,
                cond=cond,
                z=z,
                clip_denoised="dynamic",
                eta=0.,
                guidance_weight=0.,
            )[str(model.pred_steps)]

            # x_T sample from N~(0, I)
            x_t = torch.randn(x_t.shape, device=x_t.device)
            ddim_recon_randn = model.spaced_diffusion.ddim_sample(
                x_t=x_t,
                cond=cond,
                z=z,
                clip_denoised="dynamic",
                eta=0.,
                guidance_weight=0.,
            )[str(model.pred_steps)]
            return ddim_recon, ddim_recon_randn, img

        else:
            raise NotImplemented(f'callback for {model.train_pattern} is not implemented')

    @staticmethod
    def _save_to_local_and_tb(logger: pl_loggers.TensorBoardLogger,
                              global_step: int,
                              imgs: torch.Tensor,
                              tb_postfix,
                              save_path: str,
                              nrow: int = 8,
                              sample_from: str = 'target',
                              norm: bool = True) -> object:
        """
        save images to disk and tensorboard.
        :param logger:
        :param global_step:
        :param imgs:
        :param tb_postfix:
        :param save_path:
        :param nrow:
        :param sample_from:
        :param norm:
        :return:
        """
        if imgs is None:
            return
        assert len(imgs.shape) == 4, f"images should have 4 dims, but have {len(imgs.shape)}"
        grid_imgs: torch.Tensor = make_grid(imgs, nrow, normalize=norm)
        img_tensor = (grid_imgs + 1) / 2
        logger.experiment.add_image(f'samples_from_{sample_from}/{tb_postfix}', img_tensor, global_step)
        ndarray = img_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarray)
        img.save(save_path)
