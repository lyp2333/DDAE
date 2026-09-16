import torch
import os
import sys
import logging
import copy
import hydra
from typing import Union, List, Tuple
from omegaconf import DictConfig, OmegaConf

sys.path.append("/home/lyp/Code/DiffuseGAE")
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from main.models.gsl_net import Generator_fc
from main.util import configure_device, get_dataset
from lightning_fabric.utilities.seed import seed_everything
from tqdm import tqdm


@hydra.main(config_path="../../configs", config_name="default_conf")
def gsl_recombine(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(OmegaConf.to_yaml(config, resolve=True))
    config_GSL = config.dataset.GSL

    # Dataset
    root = config_GSL.data.root
    dev_ = configure_device(config_GSL.evaluation.device)
    dev = torch.device(f'cuda:{dev_[1][0]}') if isinstance(dev_, tuple) else torch.device(dev_)
    d_type = config_GSL.data.name

    image_size = config_GSL.data.image_size
    dataset = get_dataset(
        d_type, root, image_size, norm=config_GSL.data.norm, flip=config_GSL.data.hflip
    )

    # GSL-NET
    model_config = config.dataset.GSL.model
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

    # Do not put model to eval model
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=min(config_GSL.evaluation.batch_size, len(dataset)),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    # Generate sample from GSL-NET
    recon_save_path = config_GSL.evaluation.recon_save_path
    num_samples = config_GSL.evaluation.num_samples
    total = 0
    with tqdm(total=num_samples) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                batch = batch.to(dev)
                batch_size = batch.shape[0]
                recon, z = model(batch)
                batch = (batch + 1) / 2
                recon = (recon + 1) / 2
                total += batch_size
                start_index = total + 1
                for i in range(batch_size):
                    save_image(batch[i], os.path.join(recon_save_path, 'original', f'{i + start_index}'.zfill(5) + '.png'), format='png')
                    save_image(recon[i], os.path.join(recon_save_path, 'recon', f'{i + start_index}'.zfill(5) + '.png'), format='png')
                pbar.update(batch_size)
                if total >= num_samples:
                    break


if __name__ == '__main__':
    gsl_recombine()
