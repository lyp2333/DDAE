from .unet_openai import UNetModel, SuperResModel, AdaptiveUNetModel, Encoder
from .ddpm import DDPM
from .spaced_diff import SpacedDiffusion
from main.util import space_timesteps


def make_sampler(
    decoder,
    pred_steps,
    resample_strategy="spaced",
    sample_method="ddim",
    var_type='fixedsmall'
):
    # all new samplers are T with 1000, linear beta_scheduler from 1e-4 to 0.02,
    base_sampler = DDPM(
        decoder=decoder,
        var_type=var_type,
    )
    if resample_strategy == 'spaced':
        num_steps = pred_steps if pred_steps is not None else base_sampler.T
        if sample_method == 'ddpm':
            assert pred_steps == base_sampler.T
        indices = space_timesteps(base_sampler.T, num_steps)
        space_sampler = SpacedDiffusion(base_sampler, indices)
        return space_sampler
    else:
        if sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
    return base_sampler
