import shutil

import torch
import os
import sys
import logging
import copy
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

sys.path.append("/home/lyp/Code/DiffuseGAE")
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from main.models.gsl_net import Generator_fc
from main.datasets.eval import ImageDataset
from main.util import configure_device, linear_interpolation
from lightning_fabric.utilities.seed import seed_everything
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path='../../configs', config_name='default_conf.yaml')
def gsl_interpolation(config):
    logger.info(OmegaConf.to_yaml(config))
    config_joint = config.dataset.joint  # for dataset
    config_GSL = config.dataset.GSL  # for model GZS-Net

    # model_wrapper should have encoder+GAE+ddpm

    # Set seed
    seed_everything(config_joint.evaluation.seed)

    # Dataset
    # folder should be absolute path
    folder = config_joint.data.imgs_root
    assert Path(folder).is_absolute()

    dev_ = configure_device(config_joint.evaluation.device)
    dev = dev_[1] if isinstance(dev_, tuple) else dev_

    dev = torch.device(f'cuda:{dev[0]}' if isinstance(dev, list) else dev)
    d_type = config_joint.data.name
    image_size = config_joint.data.image_size
    if 'ilab' in d_type:
        attr = ['id', 'back', 'pose']
        attr_num = len(attr)
        dim_separator = [slice(0, 60),
                         slice(60, 80),
                         slice(80, 100)]

    elif 'fonts' in d_type:
        # todo change
        attr = []
        attr_num = len(attr)
        dim_separator = []
    else:
        raise NotImplementedError()
    dataset = ImageDataset(
        folder=folder,
        image_size=config_joint.data.image_size,
        exts=['jpg'],
        do_augment=False,
    )

    # Model
    model_config = config_GSL.model
    model = Generator_fc(
        model_config.nc,
        model_config.g_conv_dim,
        model_config.g_repeat_num,
        model_config.z_dim,
    ).to(dev)
    ckpt_path = config_GSL.evaluation.ckpt_path
    logger.info(f"Loading the pretrained models from  {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    logger.info(f"=> loaded checkpoint from {ckpt_path} ")
    # start interpolation
    loader = DataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    for path in dataset.paths:
        logger.info(path)
    num_intp = config_joint.evaluation.num_intp
    alpha = torch.linspace(0, 1, num_intp, dtype=torch.float32, device=dev)
    with torch.no_grad():
        with tqdm(loader, total=len(dataset)) as pbar:
            for idx, pair in enumerate(loader):
                # idx is which attr_idx we want to interpolate.
                pair = pair.to(dev)
                z_dis = model.get_latent_code(pair)
                # interpolate between z_dis
                l_z, r_z = z_dis[0], z_dis[1]
                if idx <= 2:
                    if 'ilab' in d_type:
                        # 3 groups in total
                        intp_z_each_attr = []
                        for i in range(attr_num):
                            intp_z_each_attr.append(
                                linear_interpolation(l_z[dim_separator[i]], r_z[dim_separator[i]], alpha) if i == idx
                                else ((l_z[dim_separator[i]] + r_z[dim_separator[i]]) / 2)[None, :].expand(num_intp, -1)
                            )
                        intp_imgs = model.recon_from_z(torch.cat(intp_z_each_attr, dim=-1))
                        intp_imgs = ((intp_imgs + 1) / 2).cpu()
                        save_image(intp_imgs, os.path.join(folder, f'group{idx + 1}', 'GSL_interpolation.png'), nrow=num_intp)
                        logger.info(f'We got intp_img in group{idx + 1}!')
                    else:
                        raise NotImplementedError()
                else:
                    if 'ilab' in d_type:
                        # 3 groups in total
                        intp_z_dis = linear_interpolation(l_z, r_z, alpha)
                        intp_imgs = model.recon_from_z(intp_z_dis)
                        intp_imgs = ((intp_imgs + 1) / 2).cpu()
                        save_image(intp_imgs, os.path.join(folder, f'group{idx + 1}', 'GSL_interpolation.png'), nrow=num_intp)
                        logger.info(f'We got intp_img in group{idx + 1}!')
                    else:
                        raise NotImplementedError()
                pbar.update(2)


if __name__ == '__main__':
    gsl_interpolation()
    # from pathlib import Path
    #
    # path = Path('/home/lyp/Code/encoder4editing/checkpoints/e4e_cars_encode.pt')
    # ckpt = torch.load(path)
    # opts = ckpt['opts']
    #
    # opts['checkpoint_path'] = path
    # from argparse import Namespace
    # opts = Namespace(**opts)
    # net = pSp(opts)
    # net.eval()
    # print("successfully loaded!")
