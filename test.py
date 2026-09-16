import os
import sys

sys.path.extend(['/home/lyp/Code/DiffuseGAE', '/home/lyp/Code/DiffuseGAE/third_party'])
import io
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid
import logging
import torch
import hydra
import copy
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Any
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from main.models.joint_dual_model import ModelWrapperV2 as ModelWrapper
from main.util import get_dataset, get_named_beta_schedule, parse_str
from main.models.diffusion.dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper
from main.models.diffusion import DDPM, SuperResModel, UNetModel, Encoder
from lightning_fabric.utilities.seed import seed_everything


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


# Cache directories
# ------------------------------------------------------------------------------------------

_dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


#
# def decorator(fn):
#     def func(*args, **kwargs):
#         res = fn(*args, **kwargs)
#         res *= 10
#         return res
#
#     return func
#
#
# @decorator
# def a(input1, input2):
#     return input1 + input2

# class Logging(object):
#     def __init__(self, level):
#         self.level = level
#
#     def __call__(self, func):
#         def wrapper(*args, **kwargs):
#             print("[{0}]: enter {1}()".format(self.level, func.__name__))
#             func(*args, **kwargs)
#         return wrapper
#
#
# @Logging(level="TEST")
# def hello(a, b, c):
#     print(a, b, c)

# import gradio as gr
# import cv2
# def to_black(image):
#     output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return output

# @hydra.main(config_path='./main/configs', config_name='default_conf.yaml')
# def hyper_search(config: DictConfig):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)
#     logger.info(OmegaConf.to_yaml(config, resolve=True))
#
#     config_joint = config.dataset.joint
#     seed_everything(config_joint.training.seed, workers=True)
#
#     dataset = get_dataset(
#         "ilab_128",
#         "/home/lyp/Data/ilab/ilab_128.lmdb",
#         128,
#         norm=True,
#         flip=False,
#     )
#
#     loader = DataLoader(
#         dataset,
#         8,
#         num_workers=8,
#         pin_memory=True,
#         shuffle=False,
#         drop_last=True,
#     )
#     logger.info("dataset initialized")
#
#     # superres model for conditinal training
#     attn_resolutions = parse_str(config_joint.model_ddpm.attn_resolutions)
#     dim_mults = parse_str(config_joint.model_ddpm.dim_mults)
#     ddpm_type = config_joint.training.type
#
#     decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
#     decoder = decoder_cls(
#         in_channels=config_joint.data.n_channels,
#         model_channels=config_joint.model_ddpm.dim,
#         out_channels=3,
#         emb_channels=config_joint.model_ddpm.emb_channels,  # t_emb and z_emb(generated by AE)
#         num_res_blocks=config_joint.model_ddpm.n_residual,
#         attention_resolutions=attn_resolutions,
#         channel_mult=dim_mults,
#         use_checkpoint=False,
#         dropout=config_joint.model_ddpm.dropout,
#         num_heads=config_joint.model_ddpm.n_heads,
#         z_dim=config_joint.training.z_dim,
#         use_scale_shift_norm=config_joint.model_ddpm.use_scale_shift_norm,
#         use_z=config_joint.model_ddpm.use_z,
#         use_x_hat=config_joint.model_ddpm.use_x_hat,
#         use_mappingnet=config_joint.model_ddpm.use_mappingnet,
#         pred_mode=config_joint.model_ddpm.pred_mode,
#     )
#     ema_decoder = copy.deepcopy(decoder)
#     decoder.eval()
#     ema_decoder.eval()
#
#     # base sampler(DDPM) settings
#     online_ddpm = DDPM(
#         decoder,
#         beta_1=config_joint.model_ddpm.beta1,
#         beta_2=config_joint.model_ddpm.beta2,
#         T=config_joint.model_ddpm.n_timesteps,
#         var_type=config_joint.model_ddpm.var_type,
#     )
#     target_ddpm = DDPM(
#         ema_decoder,
#         beta_1=config_joint.model_ddpm.beta1,
#         beta_2=config_joint.model_ddpm.beta2,
#         T=config_joint.model_ddpm.n_timesteps,
#         var_type=config_joint.model_ddpm.var_type,
#     )
#
#     # Encoder settings
#     config_model_encoder = config_joint.model_Encoder
#     attention_resolutions = parse_str(config_model_encoder.attention_resolutions)
#     channel_mult = parse_str(config_model_encoder.channel_mult)
#     encoder = Encoder(
#         in_channels=config_model_encoder.in_channels,
#         model_channels=config_model_encoder.model_channels,
#         out_channels=config_model_encoder.out_channels,
#         num_res_blocks=config_model_encoder.num_res_blocks,
#         attention_resolutions=attention_resolutions,
#         dropout=config_model_encoder.dropout,
#         channel_mult=channel_mult,
#         conv_resample=config_model_encoder.conv_resample,
#         use_checkpoint=config_model_encoder.use_checkpoint,
#         num_heads=config_model_encoder.num_heads,
#         resblock_updown=config_model_encoder.resblock_updown
#     )
#
#     ckpt_path = "/home/lyp/Code/DiffuseGAE/checkpoints/ilab_128-diffae-form1--checkpoints/Model_weights/last.ckpt"
#     model = ModelWrapper.load_from_checkpoint(
#         checkpoint_path=ckpt_path,
#         map_location='cuda:0',
#         strict=False,
#         online_network=online_ddpm,
#         target_network=target_ddpm,
#         encoder=encoder,
#         pred_steps=20,
#     )
#     model.eval()
#     encoder = model.encoder
#
#     logger.info("model initialized")
#     save_dir = "/home/lyp/Code/adaptive_DiffuseGAE/hyper_search_predict_eps"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     beta_scheduler = torch.from_numpy(get_named_beta_schedule('linear', 1000), )
#     noise_scheduler = NoiseScheduleVP(
#         schedule='discrete',
#         betas=beta_scheduler,
#     )
#
#     for idx, batch in enumerate(loader):
#         cond = encoder(batch)
#         model_fn = model_wrapper(
#             model=model.spaced_diffusion.decoder,
#             noise_schedule=noise_scheduler,
#             model_type='noise',
#             guidance_type='classifier-free',
#             condition=cond,
#             guidance_scale=1.,
#         )
#         logger.info('model_type: noise')
#         for algorithm_type in ["dpmsolver", "dpmsolver++"]:
#             dpm_solver = DPM_Solver(
#                 model_fn=model_fn,
#                 noise_schedule=noise_scheduler,
#                 algorithm_type=algorithm_type,
#             )
#             for method in ['singlestep', 'multistep']:
#                 for order in [2, 3]:
#                     for steps in [10, 15, 20, 25, 50, 100]:
#                         for denoise in [False]:
#
#                             save_path = os.path.join(save_dir, f"batch_{idx}-{algorithm_type}-{method}-{order}-{steps}-{denoise}")
#                             if not os.path.exists(save_path):
#                                 os.makedirs(save_path)
#                             else:
#                                 continue
#
#                             logger.info(f'current: {algorithm_type} {method} {order} {steps} {denoise}')
#                             x_t_reversed = dpm_solver.inverse(
#                                 batch,
#                                 steps=steps,
#                                 order=order,
#                                 method=method,
#                                 denoise_to_zero=denoise,
#                             )
#                             x_t_random = torch.randn(batch.shape, device=batch.device)
#
#                             sample_reversed = dpm_solver.sample(
#                                 x_t_reversed,
#                                 method=method,
#                                 order=order,
#                                 steps=steps,
#                                 denoise_to_zero=denoise,
#                             )
#
#                             sample_random = dpm_solver.sample(
#                                 x_t_reversed,
#                                 method=method,
#                                 order=order,
#                                 steps=steps,
#                                 denoise_to_zero=denoise,
#                             )
#                             grid_imgs_random = (make_grid(sample_random, 4, normalize=False) + 1) / 2
#                             grid_imgs_reversed = (make_grid(sample_reversed, 4, normalize=False) + 1) / 2
#                             ndarray_random = grid_imgs_random.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#                             ndarray_reversed = grid_imgs_reversed.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#                             imgs_random = Image.fromarray(ndarray_random)
#                             imgs_reversed = Image.fromarray(ndarray_reversed)
#                             # save images
#                             imgs_random.save(os.path.join(save_path, 'random.png'))
#                             imgs_reversed.save(os.path.join(save_path, 'reversed.png'))
#
#         break


from scipy.stats import skew
import matplotlib.pyplot as plt


def calculate_sample_skewness(sample):
    """
    计算给定样本的偏度。

    参数:
    - sample: 样本数据（NumPy 数组）

    返回:
    - skewness: 样本的偏度
    """
    assert len(sample.shape) == 2, f'samples should have 2 dims'
    mean = np.mean(sample)
    std = np.std(sample, ddof=1, axis=1, dtype=np.float64)  # 使用 ddof=1 计算样本标准差

    # 计算样本偏度
    skewness = np.mean((sample - mean) ** 3, axis=1) / np.power(std, 1.5)
    return skewness


def monte_carlo_skewness(sample_size, num_simulations=1000):
    """
    使用 Monte Carlo 方法模拟计算样本偏度的分布。

    参数:
    - sample_size: 每次模拟生成的样本大小
    - num_simulations: Monte Carlo 模拟次数

    返回:
    - skewness: 每次模拟中计算的样本偏度值列表
    """
    # 生成服从正态分布的随机样本（可以更改为其他分布）
    sample = np.random.normal(loc=0, scale=1, size=(num_simulations, sample_size))
    # 计算样本偏度
    skewness = calculate_sample_skewness(sample)

    return skewness


if __name__ == '__main__':
    # interface = gr.Interface(fn=to_black, inputs="image", outputs="image")
    # interface.launch()
    # with open("/home/ubuntu/.cache/dnnlib/downloads/2f86d6685359f59cdb0a2fab8c8f7f10_vgg16.pkl", 'rb') as f:
    #     vgg16 = pickle.load(f)
    #     print(type(vgg16))
    # lpips_fn = lpips.LPIPS(net='alex')
    # Logging('TEST')(hello)("hello,", "good", "morning")
    # GAE_PL_ckpt = torch.load("/home/lyp/Code/DiffuseGAE/checkpoints/ilab_128-l1-norm_True-500k-new_GAE-training--checkpoints/Model_weights/last.ckpt", map_location='cpu')
    # print(GAE_PL_ckpt['pytorch-lightning_version'])
    # gae_ckptt = GAE_PL_ckpt['state_dict']
    # for key in gae_ckptt:
    #     print(key)
    # print('hello')
    # print(-10 - 4 * int(-10 / 4))
    # lpath = '/home/lyp/Code/DiffuseGAE/main/evaluation/imgs_interpolation/fonts_128/group7/l.png'
    # rpath = '/home/lyp/Code/DiffuseGAE/main/evaluation/imgs_interpolation/fonts_128/group7/r.png'
    # if os.path.exists(lpath) and os.path.exists(rpath):
    #     os.remove(lpath)
    #     os.remove(rpath)
    #
    # letter = 'S'
    # src1 = f'/home/lyp/Fonts/fonts/{letter}/small/Yellow/silver/waree/{letter}_small_Yellow_silver_waree.png'
    # src2 = f'/home/lyp/Data/Fonts/fonts/{letter}/large/red/silver/waree/{letter}_large_red_silver_waree.png'
    # shutil.copy(src1, lpath)
    # shutil.copy(src2, rpath)
    # print(list(range(0, 1000, 20)))
    # hyper_search()
    # a = torch.ones((128, 128, 256))
    # b = torch.tensor((0, 2, 128))
    # a = torch.tensor(((2, 1, 3, 0, 5, 4), (0, 1, 2, 3, 4, 5)))
    # b = torch.argsort(a, dim=-1)
    # print(b)
    # from sympy.abc import a, b, c, x, y, lamda
    # from sympy import *
    # y = (x + 1) ** 2
    # print(y.expand())
    from torch import nn

    # best_acc = torch.tensor([[1., 1 / 111, 1 / 10], [1 / 6, 1., 1 / 10], [1 / 6, 1 / 111, 1.]], dtype=torch.float64)
    # dgae_acc = torch.tensor([[0.96, 0.019, 0.168], [0.15, 1.0, 0.14], [0.12, 0.015, 0.99]], dtype=torch.float64)
    # gzs_acc = torch.tensor([[.93, .020, .17], [.133, 1.0, .167], [.11, .019, .97]], dtype=torch.float64)
    # diffae_acc = torch.tensor([[.90, .018, .167], [.132, .91, .167], [.10, .019, .93]], dtype=torch.float64)
    # aeds_acc = torch.tensor([[.97, .015, .16], [.14, 1.0, .142], [.14, .016, .99]], dtype=torch.float64)
    # acc_list = [dgae_acc, diffae_acc, gzs_acc, aeds_acc]
    # for model_acc in acc_list:
    # #     res = torch.mean(torch.abs(best_acc-model_acc))
    # #     print(res)
    # a = [0, 1, 2, 3, 4]
    # print(a[-1:])
    # 参数设置
    sample_sizes = [10, 50, 100, 500, 1000]  # 每次模拟的样本容量
    num_simulations = 100000  # Monte Carlo 模拟次数
    t_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    p_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    # 进行 Monte Carlo 模拟
    for sample_size in sample_sizes:
        print(f'此时样本数量为{sample_size}')
        skewness_values = monte_carlo_skewness(sample_size, num_simulations)

        print(f"Monte Carlo 模拟估计的样本偏度均值: {np.mean(skewness_values):.4f}")
        print(f"Monte Carlo 模拟估计的样本偏度标准差: {np.std(skewness_values):.4f}")
        for t in t_list:
            print(f'小于{t}的概率为:{np.mean(skewness_values < t)}')
            print('\n\n')
        for p in p_list:
            print(f'概率为{p}的分位点为:{np.quantile(skewness_values, p)}')
            print('\n\n')
        # 结果展示

        print('\n\n\n')
    # # 绘制样本偏度的分布
    # plt.hist(skewness_values, bins=30, color='skyblue', edgecolor='black')
    # plt.xlabel('Sample Skewness')
    # plt.ylabel('Frequency')
    # plt.title('Monte Carlo Simulation of Sample Skewness')
    # plt.grid(True)
    # plt.show()
