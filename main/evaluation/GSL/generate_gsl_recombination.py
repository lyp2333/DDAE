import torch
import os
import sys
import logging
import copy
import hydra

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

sys.path.append("/home/lyp/Code/DiffuseGAE")
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from main.models.gsl_net import Generator_fc
from main.util import configure_device
from main.datasets.eval import RecombineDataset
from lightning_fabric.utilities.seed import seed_everything
from tqdm import tqdm


@hydra.main(config_path="../../configs", config_name="default_conf")
def generate_recons(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(OmegaConf.to_yaml(config, resolve=True))
    config_GSL = config.dataset.GSL

    # Set seed
    seed_everything(config_GSL.evaluation.seed)

    # Dataset
    folder = config_GSL.data.imgs_root
    assert Path(folder).is_absolute()
    dev_ = configure_device(config_GSL.evaluation.device)
    dev = torch.device(f'cuda:{dev_[1][0]}') if isinstance(dev_, tuple) else torch.device(dev_)
    d_type = config_GSL.data.name
    dim_separator = []
    if 'ilab' in d_type:
        attr = ['id', 'background', 'pose']
        attr_num = len(attr)
        dim_separator = [slice(0, 60),
                         slice(60, 80),
                         slice(80, 100)]

    elif 'fonts' in d_type:
        # todo change
        for i in range(0, 100, 20):
            dim_separator.append(slice(i, i + 20))
    else:
        raise NotImplementedError()

    # get groups
    # todo change
    dirname = 'selected_dataset'
    image_size = config_GSL.data.image_size
    dataset = RecombineDataset(
        folder=folder,
        image_size=image_size,
        dir_name=dirname,
        do_augment=False,
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
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    # recombine sample from groups via GZS-NET
    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                batch = batch.flatten(0, 1).to(dev)
                z_dis = model.get_latent_code(batch)
                # modify pose_img
                z_recombine = torch.ones(1, z_dis.shape[-1]).to(dev)
                id_z = z_dis[0]
                back_z = z_dis[1]
                pose_z = z_dis[2]
                z_recombine[0, dim_separator[0]] = id_z[dim_separator[0]]
                z_recombine[0, dim_separator[1]] = back_z[dim_separator[1]]
                z_recombine[0, dim_separator[2]] = id_z[dim_separator[2]]
                z = torch.cat((z_dis, z_recombine), dim=0)
                recon = model.recon_from_z(z)

                recon = (recon + 1) / 2
                save_image(recon, os.path.join(dataset.groups_path[idx], "replace_back_in_id_img-GZS_Net.png"), format='png', nrow=attr_num + 1)
                pbar.update(1)


if __name__ == '__main__':
    generate_recons()
