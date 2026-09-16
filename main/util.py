import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from lpips import LPIPS
from PIL import Image
from enum import Enum
from torch import distributed
from torch import nn
from torch.nn import functional as F
from .datasets import (
    FontLmdb,
    GroupFontsLmdb,
    Ilab128Lmdb,
    GroupIlabLmdb,
    GroupZIlab,
    Ilab128Lmdb_with_labels,
    Celeba64Lmdb,
    Celebahq256Lmdb,
    FFHQ256Lmdb,
)


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup if self.warmup > 0 else 1


def parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


def configure_device(device):
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = -1
        gpu_id = [int(id) for id in gpu_id.split(",")]
        return f"cuda:{gpu_id}", gpu_id
    return device


def space_timesteps(num_timesteps, desired_count, type="uniform"):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :return: a set of diffusion steps from the original process to use.
    """
    if type == "uniform":
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired_count:
                return range(0, num_timesteps, i)
        raise ValueError(
            f"cannot create exactly {desired_count} steps with an integer stride"
        )
    elif type == "quad":
        seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), desired_count) ** 2
        seq = [int(s) for s in list(seq)]
        return seq
    else:
        raise NotImplementedError


def get_dataset(name, root, image_size, dsize=-1, norm=True, flip=False, **kwargs):
    if name == "fonts_128":
        dataset = FontLmdb(
            path=root,
            image_size=image_size,
            as_tensor=True,
            do_augment=flip,
            do_norm=norm,
        )
    elif name == "fonts_128_group":
        dataset = GroupFontsLmdb(
            path=root,
            group_size=6,
            image_size=image_size,
            as_tensor=True,
            do_augment=flip,
            do_normalize=norm,
        )
    elif name == 'fonts_128_group_20k_z':
        dataset = GroupZIlab(
            path=root,
            do_normalize=norm,
            dataset_size=dsize,
        )
    elif name == "ilab_128":
        dataset = Ilab128Lmdb(
            path=root,
            image_size=image_size,
            as_tensor=True,
            do_augment=flip,
            do_norm=norm,
        )
    elif name == 'ilab_128_group':
        dataset = GroupIlabLmdb(
            path=root,
            image_size=image_size,
            group_size=4,
            as_tensor=True,
            do_augment=flip,
            do_normalize=norm,
        )
    elif name == 'ilab_128_group_z':
        dataset = GroupZIlab(
            path=root,
            do_normalize=norm,
            dataset_size=dsize,
        )

    elif name == "ilab_128_for_classify":
        dataset = Ilab128Lmdb_with_labels(
            path=root,
            do_norm=norm,
            do_augment=False,
        )
    elif name == "celeba_64":
        dataset = Celeba64Lmdb(
            path=root,
            image_size=image_size,
            as_tensor=True,
            do_norm=norm,
            do_augment=flip,
        )
    elif name == "celebahq_256":
        dataset = Celebahq256Lmdb(
            path=root,
            image_size=256,
            as_tensor=True,
            do_norm=norm,
            do_augment=flip,
        )
    elif name == "ffhq_256":
        dataset = FFHQ256Lmdb(
            path=root,
            image_size=256,
            as_tensor=True,
            do_norm=norm,
            do_augment=flip,
        )
    else:
        raise NotImplementedError(
            f"The dataset {name} does not exist in our datastore."
        )
    return dataset


def convert_to_np(obj):
    obj = obj.permute(0, 2, 3, 1).contiguous()
    obj = obj.detach().cpu().numpy()

    obj_list = []
    for _, out in enumerate(obj):
        obj_list.append(out)
    return obj_list


def normalize(obj):
    B, C, H, W = obj.shape
    for i in range(3):
        channel_val = obj[:, i, :, :].view(B, -1)
        channel_val -= channel_val.min(1, keepdim=True)[0]
        channel_val /= (
            channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0]
        )
        channel_val = channel_val.view(B, H, W)
        obj[:, i, :, :] = channel_val
    return obj


def save_as_images(obj, file_name="output", denorm=True):
    # Saves predictions as png images (useful for Sample generation)
    if denorm:
        # obj = normalize(obj)
        obj = obj * 0.5 + 0.5
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        out = (out * 255).clip(0, 255).astype(np.uint8)
        img_out = Image.fromarray(out)
        current_file_name = file_name + f"_{i}.png"
        img_out.save(current_file_name, "png")


def save_as_np(obj, file_name="output", denorm=True):
    # Saves predictions directly as numpy arrays
    if denorm:
        obj = normalize(obj)
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        current_file_name = file_name + f"_{i}.npy"
        np.save(current_file_name, out)


def compare_samples(samples, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    ncols = len(samples)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    for idx, (caption, img) in enumerate(samples.items()):
        ax[idx].imshow(img.permute(1, 2, 0))
        ax[idx].set_title(caption)
        ax[idx].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, pad_inches=0)

    plt.close()


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start,
                           beta_end,
                           num_diffusion_timesteps,
                           dtype=np.float64)
    # cumulative alpha_t follows the cosine strategy
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "const0.01":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.01] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.015] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0065":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0065] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0055":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0055] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0045":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0045] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0035":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0035] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0025":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0025] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0015] * num_diffusion_timesteps,
                        dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    alpha value for discretized alpha_bar (t in [0, 1])
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def get_rank():
    if distributed.is_initialized():
        return distributed.get_rank()
    else:
        return 0


def get_world_size():
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def linear_interpolation(l_z, r_z, alpha: torch.Tensor):
    assert l_z.shape == r_z.shape, f'l_z.shape: {l_z.shape}, r_z.shape: {r_z.shape}'
    if len(l_z.shape) == 1:
        intp_z = l_z[None, :] * (1 - alpha[:, None]) + r_z[None, :] * alpha[:, None]
    else:
        assert l_z.size(0) == alpha.size(0)
        intp_z = l_z * (1 - alpha) + r_z * alpha
    return intp_z


def _cos(a: torch.Tensor, b: torch.Tensor):
    a = F.normalize(a.flatten(1, 3), p=2, dim=1)
    b = F.normalize(b.flatten(1, 3), p=2, dim=1)
    return (a * b).sum(dim=1)


def sphere_interpolation(l_img, r_img, alpha: torch.Tensor):
    assert l_img.shape == r_img.shape
    alpha = alpha if len(alpha.shape) == 2 else alpha[None, ...]
    if len(l_img.shape) == 3:
        l_img, r_img = l_img[None, ...], r_img[None, ...]
    img_shape = l_img.shape
    theta = torch.arccos(_cos(l_img, r_img))[..., None]

    intp_x = ((torch.sin((1 - alpha) * theta)[..., None] * l_img.flatten(1, 3)[:, None, :] +
               torch.sin(alpha * theta)[..., None] * r_img.flatten(1, 3)[:, None, :])
              / torch.sin(theta)[..., None]).reshape(img_shape[0], -1, img_shape[1], img_shape[2], img_shape[3]).squeeze()

    return intp_x


def dynamic_thresholding_fn(x0,
                            dynamic_thresholding_ratio=0.995,
                            thresholding_max_val=1.,
                            ):
    """
    The dynamic thresholding method.
    """
    dims = x0.dim()
    p = dynamic_thresholding_ratio
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = expand_dims(torch.maximum(s, thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
    x0 = torch.clamp(x0, -s, s) / s
    return x0


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# def sphere_interpolation_single_step_(a, b, t):
#     assert a.shape == b.shape
#     a = a / a.norm(dim=-1, keepdim=True)
#     b = b / b.norm(dim=-1, keepdim=True)
#     d = (a * b).sum(dim=-1, keepdim=True)
#     p = t * torch.acos(d)
#     c = b - d * a
#     c = c / c.norm(dim=-1, keepdim=True)
#     d = a * torch.cos(p) + c * torch.sin(p)
#     d = d / d.norm(dim=-1, keepdim=True)
#     return d


def get_pair_index(batch_size):
    odd_index = []
    even_index = []
    for i in range(batch_size):
        if i % 2 == 0:
            even_index.append(i)
        else:
            odd_index.append(i)
    return odd_index, even_index


# Enum Class
class ClassifierTrainMode(Enum):
    encoder_out = 'encoder_out'
    gae_out = 'gae_out'
    gsl_out = 'gsl_out'

    def is_using_whole_z(self):
        return self in [
            ClassifierTrainMode.encoder_out
        ]


class LPIPSwithPixel(nn.Module):
    def __init__(
        self,
        lpips_model="vgg",
        pixel_loss="l2",
        lpips_loss_weight=1.0,
        pixel_loss_weight=1.0,
        normalize_lpips=True,
    ):
        super(LPIPSwithPixel, self).__init__()
        self.lpips_f = LPIPS(net=lpips_model, pretrained=True)
        self.pixel_f = nn.L1Loss() if pixel_loss == "l1" else nn.MSELoss()
        self.lpips_weight = lpips_loss_weight
        self.pixel_weight = pixel_loss_weight
        self.normlize_lpips = normalize_lpips

    def forward(self, input1, input2):
        lpips_loss = self.lpips_f.forward(input1, input2, normalize=self.normlize_lpips) if self.lpips_weight > 0 else 0
        pixel_loss = self.pixel_f(input1, input2) if self.pixel_weight > 0 else 0
        return (self.lpips_weight * lpips_loss + self.pixel_weight * pixel_loss).mean()
