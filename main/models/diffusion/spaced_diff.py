import torch.nn as nn
import torch
from functools import partial
from main.util import dynamic_thresholding_fn, extract


class SpacedDiffusion(nn.Module):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    """

    def __init__(
        self,
        base_diffusion,
        use_timesteps,
    ):
        super().__init__()
        self.base_diffusion: nn.Module = base_diffusion
        self.use_timesteps: list = use_timesteps
        self.original_num_steps: int = self.base_diffusion.T
        self.decoder: nn.Module = self.base_diffusion.decoder
        self.use_adaptive = self.decoder.use_adaptive
        self.pred_mode = self.decoder.pred_mode
        self.var_type: str = self.base_diffusion.var_type
        self.timestep_map = []

        # ---- Implicit constraint (paper §3.3) configuration ----
        # `implicit_constraint_t_frac`: fraction of late denoising steps
        #     at which K/V re-injection is active. Default 0.1 matches
        #     "last 10% of denoising steps" (supp.tex §4).
        # `_decoder_kv_inject_on`: internal flag tracking whether the
        #     decoder is currently in injection mode, so we don't toggle
        #     `enable_implicit_constraint_generation` / `disable_*` on
        #     every step.
        self.implicit_constraint_t_frac: float = 0.1
        self._decoder_kv_inject_on: bool = False

        last_alpha_cumprod = 1.0
        alphas_cumprod = torch.cumprod(1.0 - self.base_diffusion.betas, dim=0)
        new_betas = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(torch.tensor([1 - alpha_cumprod / last_alpha_cumprod]))
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        self.register_buffer("betas", torch.cat(new_betas))
        dev = self.betas.device
        self.alphas = 1.0 - self.betas
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / self.alphas))
        self.register_buffer("alpha_bar", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alpha_bar_shifted",
            torch.cat([torch.tensor([1.0], device=dev), self.alpha_bar[:-1]]),
        )
        self.register_buffer(
            "alpha_bar_next",
            torch.cat((self.alpha_bar[1:], torch.tensor([0.0], device=dev)))
        )

        assert self.alpha_bar_shifted.shape == torch.Size(
            [
                len(self.timestep_map),
            ]
        )

        # Auxillary consts
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer("minus_sqrt_alpha_bar", torch.sqrt(1.0 - self.alpha_bar))
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alpha_bar)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alpha_bar - 1)
        )

        # Posterior q(x_t-1|x_t,x_0,t) covariance of the forward process
        self.register_buffer(
            "post_variance",
            self.betas * (1.0 - self.alpha_bar_shifted) / (1.0 - self.alpha_bar),
        )
        # Clipping because post_variance is 0 before the chain starts
        self.register_buffer(
            "post_log_variance_clipped",
            torch.log(
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=dev),
                        self.post_variance[1:],
                    ]
                )
            ),
        )

        # q(x_t-1 | x_t, x_0) mean coefficients
        self.register_buffer(
            "post_coeff_1",
            self.betas * torch.sqrt(self.alpha_bar_shifted) / (1.0 - self.alpha_bar),
        )
        self.register_buffer(
            "post_coeff_2",
            torch.sqrt(self.alphas) * (1 - self.alpha_bar_shifted) / (1 - self.alpha_bar),
        )
        # q(x_t-1 | x_t, eps) mean coefficients
        self.register_buffer(
            "post_coeff_3",
            self.betas / self.minus_sqrt_alpha_bar,
        )

    def _predict_xstart_from_eps(self, x_t, t_, eps):
        assert x_t.shape == eps.shape
        _x_start = extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) * x_t \
                   - extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape) * eps
        return _x_start

    def _predict_eps_from_xstart(self, x_t, t_, pred_xstart):
        return (extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) *
                x_t - pred_xstart) / extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape)

    def get_posterior_mean_covariance_x0(
        self, x_t, t, clip_denoised=True, cond=None, z=None, guidance_weight=0.0, **kwargs
    ):
        """
        get post_mean, variance, pred_start from x_t
        :param x_t:
        :param t:
        :param clip_denoised:
        :param cond:
        :param z:
        :param guidance_weight:
        :param kwargs:
        :return:
        """
        B = x_t.size(0)
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        t_model_ = torch.full(
            (x_t.size(0),), self.timestep_map[t], device=x_t.device, dtype=torch.long
        )
        assert t_.shape == torch.Size(
            [
                B,
            ]
        )

        if guidance_weight == 0:
            eps = self.decoder(x_t, t_model_, low_res=cond, z=z)
        else:
            eps = (1 + guidance_weight) * self.decoder(
                x_t, t_model_, low_res=cond, z=z
            ) - guidance_weight * self.decoder(
                x_t,
                t_model_,
                low_res=torch.zeros_like(cond),
                z=torch.zeros_like(z) if z is not None else None,
            )

        # Generate the reconstruction from x_t
        x_recons = self._predict_xstart_from_eps(x_t, t_, eps)

        # Clip
        if clip_denoised:
            x_recons.clamp_(-1.0, 1.0)

        # Compute posterior mean from the reconstruction
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x_recons
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
        )

        # Extract posterior variance
        p_variance, p_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            "fixedlarge": (
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=x_t.device),
                        self.betas[1:],
                    ]
                ),
                torch.log(
                    torch.cat(
                        [
                            torch.tensor([self.post_variance[1]], device=x_t.device),
                            self.betas[1:],
                        ]
                    )
                ),
            ),
            "fixedsmall": (
                self.post_variance,
                self.post_log_variance_clipped,
            ),
        }[self.var_type]
        post_variance = extract(p_variance, t_, x_t.shape)
        post_log_variance = extract(p_log_variance, t_, x_t.shape)
        return post_mean, post_variance, post_log_variance, x_recons, eps

    def get_posterior_mean_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, _eps_score=None, cond=None, z=None):
        assert _eps_score is not None
        # re-derive posterior_mean to get mean from x_t and eps directly
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long) if isinstance(t, int) else t
        # t_model_ = torch.full((x_t.size(0),), self.timestep_map[t], device=x_t.device) if isinstance(t, int) \
        #     else torch.gather(torch.tensor(self.timestep_map), 0, t_).to(x_t.device)
        # get real eps_score
        # if _eps_score is None:
        #     eps_score = self.decoder(x_t, t_model_, low_res=cond, z=z)[..., :3, :, :] if self.use_adaptive else self.decoder(x_t, t_model_, low_res=cond, z=z)
        # else:
        eps_score = _eps_score

        # Generate the reconstruction from x_t and eps
        post_mean = extract(self.sqrt_recip_alpha, t_, x_t.shape) * \
                    (x_t - extract(self.post_coeff_3, t_, x_t.shape) * eps_score)
        return post_mean

    def get_posterior_mean_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, _x0_score=None, clip_denoised=True, cond=None, z=None):
        assert _x0_score is not None
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long) if isinstance(t, int) else t
        # t_model_ = torch.full((x_t.size(0),), self.timestep_map[t], device=x_t.device) if isinstance(t, int) \
        #     else torch.gather(torch.tensor(self.timestep_map), 0, t_).to(x_t.device)

        # get real x0_score
        # if _x0_score is None:
        #     x0_score = self.decoder(x_t, t_model_, low_res=cond, z=z)[..., 3:6, :, :] if self.use_adaptive else self.decoder(x_t, t_model_, low_res=cond, z=z)
        # else:
        x0_score = _x0_score

        # clip x0_score
        if clip_denoised:
            # todo add dynamic threshold
            x0_score.clamp_(-1.0, 1.0)

        # Generate the reconstruction from x_t and x_0
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x0_score
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
        )
        return post_mean

    def get_ddim_mean_cov(
        self,
        x,
        t,
        clip_denoised="static",
        cond=None,
        z=None,
        eta=0.0,
        guidance_weight=0.0,
    ):
        B = x.size(0)
        t_ = torch.full((x.size(0),), t, device=x.device)
        t_model_ = torch.full(
            (x.size(0),), self.timestep_map[t], device=x.device
        )
        assert t_.shape == torch.Size(
            [
                B,
            ]
        )
        # Generate the reconstruction from x_t
        if guidance_weight == 0:
            pred = self.decoder(x,
                                t_model_,
                                low_res=cond if cond is not None else None,
                                z=z if z is not None else None,
                                )

        else:
            # todo add classifier free guidance
            # eps = (1 + guidance_weight) * self.decoder(
            #     x,
            #     t_model_,
            #     low_res=cond if cond is not None else None,
            #     z=z if z is not None else None
            # ) - guidance_weight * self.decoder(
            #     x,
            #     t_model_,
            #     low_res=torch.zeros_like(cond) if cond is not None else None,
            #     z=torch.zeros_like(z) if z is not None else None,
            # )
            ...

        x_recons = self._predict_xstart_from_eps(x, t_, pred) if self.pred_mode == 'eps' else pred
        eps = self._predict_eps_from_xstart(x, t_, pred) if self.pred_mode == 'x0' else pred
        # Clip
        if clip_denoised == "static":
            x_recons.clamp_(-1.0, 1.0)
        elif clip_denoised == "dynamic":
            x_recons = dynamic_thresholding_fn(x_recons)
        else:
            raise NotImplementedError('method is not implemented for clipping x_start')
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction. we re-derive eps with p_mean_variance
        # eps = self._predict_eps_from_xstart(x, t_, x_recons)
        # get mean and variance
        alpha_bar = extract(self.alpha_bar, t_, x.shape)
        alpha_bar_prev = extract(self.alpha_bar_shifted, t_, x.shape)
        # eta == 0, ddim is a deterministic process, eta = 1, forward process becomes markovian, ddim becomes a DDPM
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        mean_pred = (
            x_recons * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        return mean_pred, sigma

    def get_adaptive_ddim_mean_cov(
        self,
        x_t,
        t,
        adaptive_factor_=None,
        clip_denoised="static",
        cond=None,
        z=None,
        eta=0.0,
        guidance_weight=0.0,
    ):
        """
        get ddim_mean_cov and from adaptive decoder(Unet)
        :param x:
        :param t:
        :param adaptive_factor_:
        :param clip_denoised:
        :param cond:
        :param z:
        :param eta:
        :param guidance_weight:
        :return:
        """
        B = x_t.size(0)
        t_ = torch.full((x_t.size(0),), t, device=x_t.device)
        t_model_ = torch.full(
            (x_t.size(0),), self.timestep_map[t], device=x_t.device
        )
        assert t_.shape == torch.Size(
            [
                B,
            ]
        )
        alpha_bar = extract(self.alpha_bar, t_, x_t.shape)
        alpha_bar_prev = extract(self.alpha_bar_shifted, t_, x_t.shape)
        # we only change the mean_pred, variance is the same
        # eta == 0, ddim is a deterministic process, eta = 1, forward process becomes markovian, ddim becomes a DDPM
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Generate the reconstruction from x_t
        if guidance_weight == 0:
            pred_score = self.decoder(x_t,
                                      t_model_,
                                      low_res=cond if cond is not None else None,
                                      z=z if z is not None else None,
                                      )
            eps_score, x0_score = pred_score[..., :3, :, :], pred_score[..., 3:6, :, :]
            adaptive_factor = pred_score[..., 6:7, :, :] if adaptive_factor_ is None else adaptive_factor_
            if clip_denoised == "static":
                x0_score.clamp_(-1.0, 1.0)
            elif clip_denoised == "dynamic":
                x0_score = dynamic_thresholding_fn(x0_score)
            else:
                raise NotImplementedError('method is not implemented for clipping x_start')

            if self.pred_mode == "adaptive_form1":

                # Generate the posterior from noise-pred and x_0-pred
                # noise-pred
                x_recons = self._predict_xstart_from_eps(x_t, t_, eps_score)
                # Clip
                if clip_denoised == "static":
                    x_recons.clamp_(-1.0, 1.0)
                elif clip_denoised == "dynamic":
                    x_recons = dynamic_thresholding_fn(x_recons)
                else:
                    raise NotImplementedError('method is not implemented for clipping x_start')

                alpha_bar = extract(self.alpha_bar, t_, x_t.shape)
                alpha_bar_prev = extract(self.alpha_bar_shifted, t_, x_t.shape)
                # eta == 0, ddim is a deterministic process, eta = 1, forward process becomes markovian, ddim becomes a DDPM
                sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                pred_xt_minus1_from_noise = (x_recons * torch.sqrt(alpha_bar_prev) +
                                             torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_score)

                # x0_pred
                eps = self._predict_eps_from_xstart(x_t, t_, x0_score)
                pred_xt_minus1_from_x0 = (x0_score * torch.sqrt(alpha_bar_prev) +
                                          torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)

                mean_pred = adaptive_factor * pred_xt_minus1_from_noise + \
                            (1 - adaptive_factor) * pred_xt_minus1_from_x0
                return mean_pred, 0
            elif self.pred_mode == "adaptive_form2":
                modified_eps_score = (1 - adaptive_factor) * self._predict_eps_from_xstart(x_t, t_, x0_score) \
                                     + adaptive_factor * eps_score
                modified_x0_score = self._predict_xstart_from_eps(x_t, t_, modified_eps_score)

                # Clip
                if clip_denoised == "static":
                    modified_x_recons = modified_x0_score.clamp(-1.0, 1.0)
                elif clip_denoised == "dynamic":
                    modified_x_recons = dynamic_thresholding_fn(modified_x0_score)
                else:
                    raise NotImplementedError('method is not implemented for clipping x_start')

                # Equation 12.
                mean_pred = (
                    modified_x_recons * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * modified_eps_score
                )
                return mean_pred, sigma
            else:
                raise NotImplementedError()
        else:
            # todo change, get two types of pred and interpolate them via adaptive factor(add sigma * \mathhcal{N}(0,I) in ddpm and without in ddim)
            # pred = (1 + guidance_weight) * self.decoder(
            #     x_t,
            #     t_model_,
            #     low_res=cond if cond is not None else None,
            #     z=z if z is not None else None
            # ) - guidance_weight * self.decoder(
            #     x_t,
            #     t_model_,
            #     low_res=torch.zeros_like(cond) if cond is not None else None,
            #     z=torch.zeros_like(z) if z is not None else None,
            # )
            ...

    def ddim_sample(self,
                    x_t,
                    cond=None,
                    z=None,
                    adaptive_factor=None,
                    clip_denoised="static",
                    checkpoints=[],
                    eta=0.0,
                    guidance_weight=0.0,
                    use_implicit_constraint=False,
                    ):
        """
        ddim sample with adaptive model
        :param x_t:
        :param cond:
        :param z:
        :param adaptive_factor:
        :param clip_denoised:
        :param checkpoints:
        :param eta:
        :param guidance_weight:
        :param use_implicit_constraint: if True, enable K/V re-injection
            from the inversion pass (paper §3.3). The decoder's
            self-attention blocks will REPLACE their freshly computed
            K/V tensors with the cached ones from the inversion pass,
            for the LAST `t_star_frac` fraction of denoising steps (and
            for the LAST 3 decoder blocks by default; see also
            `paper §3.3`).
        :return:
        """
        # The sampling process w and w/o adaptive model goes here!
        x = x_t
        B, *_ = x_t.shape
        sample_dict = {}
        get_mean_cov = partial(self.get_adaptive_ddim_mean_cov, adaptive_factor_=adaptive_factor) if self.use_adaptive else self.get_ddim_mean_cov
        num_steps = len(self.timestep_map)
        checkpoints = [num_steps] if checkpoints == [] else checkpoints

        # ---- Implicit constraint (paper §3.3) ----
        # Only enable K/V injection for the late denoising steps
        # (t < t_star), where image layout is established; this matches
        # the "last 10% of denoising steps" recipe in supp.tex §4.
        implicit_constraint_active = False
        if use_implicit_constraint and hasattr(self.decoder, 'enable_implicit_constraint_generation'):
            self.decoder.enable_implicit_constraint_generation()
            implicit_constraint_active = True
            t_star = max(1, int(num_steps * self.implicit_constraint_t_frac))

        for idx, t in enumerate(reversed(range(0, num_steps))):
            # Toggle the implicit constraint based on whether we are in
            # the "high-resolution" late phase.
            if implicit_constraint_active:
                if (t < t_star) and not self._decoder_kv_inject_on:
                    self.decoder.enable_implicit_constraint_generation()
                    self._decoder_kv_inject_on = True
                elif (t >= t_star) and self._decoder_kv_inject_on:
                    self.decoder.disable_implicit_constraint()
                    self._decoder_kv_inject_on = False

            noise = torch.randn_like(x_t)
            assert noise.shape == x_t.shape
            # x += 0.002 * x_t
            (post_mean, post_variance) = get_mean_cov(
                x,
                t,
                cond=cond,
                z=z,
                clip_denoised=clip_denoised,
                eta=eta,
                guidance_weight=guidance_weight,
            )
            nonzero_mask = (
                torch.tensor(t != 0, device=x.device)
                .float()
                .view(-1, *([1] * (len(x_t.shape) - 1)))
            )  # no noise when t == 0

            # Langevin step!
            x = post_mean + nonzero_mask * post_variance * noise
            sample_dict[f'adaptive_factor-{idx + 1}'] = adaptive_factor
            # Add results
            if idx + 1 in checkpoints:
                sample_dict[str(idx + 1)] = x

        # Always restore the default attention behavior before returning
        # so subsequent passes without `use_implicit_constraint=True`
        # are not affected.
        if implicit_constraint_active:
            self.decoder.disable_implicit_constraint()
            self._decoder_kv_inject_on = False

        return sample_dict

    def ddim_reverse_sample(
        self,
        x,
        t,
        cond=None,
        z=None,
        clip_denoised=True,
        eta=0.0,
        **kwargs,
    ):
        assert eta == 0.0, f'in deterministic process, eta should be zero'
        post_mean, _, _, pred_xstart, eps = self.get_posterior_mean_covariance_x0(
            x,
            t,
            cond=cond,
            z=z,
            clip_denoised=clip_denoised,
            guidance_weight=0,
            **kwargs
        )

        t_ = torch.full((x.size(0),), t, device=x.device)
        # in case we used x_start or x_prev prediction , we re-derive eps only use x_t
        eps = (extract(self.sqrt_recip_alphas_cumprod, t_, x.shape) * x - pred_xstart) \
              / extract(self.sqrt_recipm1_alphas_cumprod, t_, x.shape)
        alpha_bar_next = extract(self.alpha_bar_next, t_, x.shape)

        mean_pred = pred_xstart * torch.sqrt(alpha_bar_next) \
                    + torch.sqrt(1 - alpha_bar_next) * eps
        return mean_pred, pred_xstart

    def ddim_reverse_sample_with_adaptive_model(
        self,
        x,
        t,
        cond=None,
        z=None,
        clip_denoised=True,
        eta=0.0,
        **kwargs,
    ):
        """
        ddim reverse sample with adaptive model
        :param x:
        :param t:
        :param cond:
        :param z:
        :param clip_denoised:
        :param eta:
        :param kwargs:
        :return:
        """
        assert eta == 0.0, f'in deterministic process, eta should be zero'
        t_ = torch.full((x.size(0),), t, device=x.device)
        t_model_ = torch.full(
            (x.size(0),), self.timestep_map[t], device=x.device
        )

        pred_score = self.decoder(x,
                                  t_model_,
                                  low_res=cond,
                                  z=z,
                                  )

        eps_score, x0_score, adaptive_factor = pred_score[..., :3, :, :], pred_score[..., 3:6, :, :], pred_score[..., 6:7, :, :]
        if self.pred_mode == "adaptive_form1":
            modified_eps_score = (extract(self.sqrt_recip_alphas_cumprod, t_, x.shape) * x - x0_score) / \
                                 extract(self.sqrt_recipm1_alphas_cumprod, t_, x.shape)

            alpha_bar_next = extract(self.alpha_bar_next, t_, x.shape)

            mean_pred = x0_score * torch.sqrt(alpha_bar_next) \
                        + torch.sqrt(1 - alpha_bar_next) * modified_eps_score
            return mean_pred, x0_score
        elif self.pred_mode == "adaptive_form2":
            modified_eps_score = (1 - adaptive_factor) * \
                                 self.base_diffusion.get_posterior_eps_from_xt_x0(
                                     x,
                                     t_model_,
                                     x0_score,
                                     clip_denoised=True,
                                     cond=None,
                                     z=None,
                                 ) \
                                 + adaptive_factor * eps_score
            modified_x0_score = self._predict_xstart_from_eps(x, t_, modified_eps_score)

            modified_eps_score = (extract(self.sqrt_recip_alphas_cumprod, t_, x.shape) * x - modified_x0_score) \
                                 / extract(self.sqrt_recipm1_alphas_cumprod, t_, x.shape)

            alpha_bar_next = extract(self.alpha_bar_next, t_, x.shape)

            mean_pred = modified_x0_score * torch.sqrt(alpha_bar_next) \
                        + torch.sqrt(1 - alpha_bar_next) * modified_eps_score
            return mean_pred, modified_x0_score
        else:
            raise NotImplementedError()

        # todo replace mentioned above
        # mean_pred = x0_score * extract(self.sqrt_alpha_bar, t, x0_score.shape) + \
        #             eps_score * extract(self.minus_sqrt_alpha_bar, t, x0_score.shape)

    def ddim_reverse_sample_loop(self, x, cond=None, z=None, clip_denoised=True, eta=0.0,
                                  use_implicit_constraint=False, **kwargs):
        """
        DDIM inversion loop.

        :param use_implicit_constraint: if True, populate the U-Net's K/V cache
            during inversion so that subsequent ``ddim_sample`` calls with the
            same flag can re-inject those K/V values (paper §3.3 implicit
            constraint).
        """
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(len(self.use_timesteps)))
        reverse_sample = self.ddim_reverse_sample_with_adaptive_model if self.use_adaptive else self.ddim_reverse_sample
        sample: torch.Tensor = x

        # Enable K/V caching on every attention block in the U-Net.  Because
        # the inversion pass is deterministic (eta=0), only the final step's
        # K/V is kept -- the cache uses a single key that is overwritten on
        # every forward pass.
        if use_implicit_constraint and hasattr(self.decoder, 'enable_implicit_constraint_inversion'):
            self.decoder.enable_implicit_constraint_inversion()
        try:
            for t in indices:
                mean_pred, pred_x_start = reverse_sample(
                    x=sample,
                    t=t,
                    cond=cond,
                    z=z,
                    clip_denoised=clip_denoised,
                    eta=eta,
                    **kwargs
                )
                sample = mean_pred
                # bugfix here, sample is tensor in first loop, tuple with one element in the next loops.
                sample = sample[0] if isinstance(sample, tuple) else sample
                sample_t.append(sample)
                xstart_t.append(pred_x_start)
                T.append(t)
        finally:
            # Always restore default behaviour, even if an exception is raised.
            if hasattr(self.decoder, 'disable_implicit_constraint'):
                self.decoder.disable_implicit_constraint()

        return {
            'sample': sample,
            'sample_t': sample_t
        }
