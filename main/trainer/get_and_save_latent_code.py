import torch
import hydra
import sys
import logging
from os import path as osp
from torch.utils.data import DataLoader
from main.util import get_dataset, parse_str, configure_device
from tqdm import tqdm
from ..models.diffusion.unet_openai import Encoder
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="../configs", config_name="default_conf.yaml")
def get_latent(config):
    config_joint = config.dataset.joint
    config_model_encoder = config_joint.model_Encoder
    dev = configure_device(config_joint.training.device)
    dev = torch.device(f'cuda:{dev[1][0]}' if isinstance(dev, tuple) else 'cpu')
    is_group_dataset = config_joint.data.is_group_dataset
    name = config_joint.data.name
    saved_name = config_joint.data.saved_name
    dataset = get_dataset(
        name=name,
        root=config_joint.data.root,
        image_size=config_joint.data.image_size,
        norm=True,
        flip=False,
    )
    DATASET_DIR = '/home/lyp/Data/ilab'
    batch_size = config_joint.training.batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config_joint.training.workers,
        drop_last=False,
    )

    # get encoder
    attention_resolutions = parse_str(config_model_encoder.attention_resolutions)
    channel_mult = parse_str(config_model_encoder.channel_mult)
    encoder = Encoder(
        in_channels=config_model_encoder.in_channels,
        model_channels=config_model_encoder.model_channels,
        out_channels=config_model_encoder.out_channels,
        num_res_blocks=config_model_encoder.num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=config_model_encoder.dropout,
        channel_mult=channel_mult,
        conv_resample=config_model_encoder.conv_resample,
        use_checkpoint=config_model_encoder.use_checkpoint,
        num_heads=config_model_encoder.num_heads,
        resblock_updown=config_model_encoder.resblock_updown
    )

    logdir_path = osp.join(config_joint.training.results_dir, 'checkpoints', f'{config_joint.training.training_name}--checkpoints')
    mode = config_joint.training.mode
    last_ckpt_path = next(Path(logdir_path + "/Model_weights").glob(f'last*'))
    best_ckpt_path = next(Path(logdir_path + "/Model_weights").glob(f'model*'))
    ckpt_path = last_ckpt_path if mode == 'last' else best_ckpt_path
    logger.info(f'load checkpoint from {ckpt_path}')
    ckpt_joint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    ckpt_encoder = {}
    for key in ckpt_joint['state_dict']:
        if 'encoder' in key:
            ckpt_encoder[key.replace('encoder.', '')] = ckpt_joint['state_dict'][key]

    encoder.load_state_dict(ckpt_encoder, strict=True)
    encoder.to(device=dev)
    encoder.eval()

    # get latent code
    all_z = []
    z_dim = config_model_encoder.out_channels
    if is_group_dataset:
        group_size, *img_attr = dataset[0].shape
    else:
        img_attr = dataset[0].shape

    with torch.no_grad():
        with tqdm(total=(len(dataset))) as pbar:
            for idx, batch in enumerate(loader):
                batch = batch.to(dev)
                batch = batch.flatten(0, 1) if is_group_dataset else batch
                z = encoder(batch).to(dev)
                all_z.append(z)
                pbar.update(batch_size)
    all_z = torch.cat(all_z, dim=0) if not is_group_dataset else torch.cat(all_z, dim=0).reshape(-1, group_size, z_dim)
    data = {}
    data['data'] = all_z
    shape = all_z.shape
    data['mean'] = torch.mean(all_z, dim=[0, 1]) if len(shape) == 3 else torch.mean(all_z, dim=[0])
    data['std'] = torch.std(all_z, dim=[0, 1]) if len(shape) == 3 else torch.std(all_z, dim=[0])

    torch.save(data, osp.join(DATASET_DIR, 'latent_code', f'group_latent_code-{saved_name}.pth' if is_group_dataset else f'latent_code-{saved_name}.pth'))


if __name__ == '__main__':
    get_latent()
