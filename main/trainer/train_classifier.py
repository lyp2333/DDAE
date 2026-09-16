import os
import sys
import time

import torch
import copy
import logging
import hydra
import pytorch_lightning as pl
from os import path as osp
from os import makedirs
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from ..models.diffusion import Encoder
from ..models.autoencoders.GAE import GAE_PL
from ..models.gsl_net import Generator_fc
from ..models.classifier import ClsModel
from main.util import configure_device, get_dataset, parse_str
from collections import OrderedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path='../configs', config_name='default_conf.yaml')
def train(config):
    config_joint = config.dataset.joint
    config_classifier = config.dataset.classifier

    logger.info(OmegaConf.to_yaml(config.dataset, resolve=True))
    # Set seed
    seed_everything(config_classifier.training.seed, workers=True)

    # Dataset
    root = config_classifier.data.root
    d_type = config_classifier.data.name
    image_size = config_classifier.data.image_size
    dataset = get_dataset(
        d_type, root, image_size, norm=config_classifier.data.norm, flip=False
    )
    N = len(dataset)
    batch_size = config_classifier.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    # GAE setting
    Gae_pl = GAE_PL.load_from_checkpoint(config_classifier.model.GAE_ckpt_path, strict=False)
    Gae_pl.eval()
    if config_classifier.model.z_norm:
        z = torch.load(config_classifier.data.z_root)
        z_mean_std = (z["mean"], z["std"])
    else:
        z_mean_std = None

    # Encoder settings
    config_model_encoder = config_joint.model_Encoder
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
    state_dict = OrderedDict()
    for key, val in torch.load(config_classifier.model.encoder_ckpt_path)['state_dict'].items():
        if 'encoder.' in key:
            state_dict[key.replace('encoder.', '')] = val
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()

    # GSL settings
    gsl_state_dict = torch.load(config_classifier.model.GSL_ckpt_path)
    gsl = Generator_fc(
        nc=3,
        conv_dim=64,
        repeat_num=1,
        z_dim=100,
    ).load_state_dict(gsl_state_dict, strict=True)

    # classifier setting
    classifier = ClsModel(
        encoder=encoder,
        gae_pl=Gae_pl,
        gsl=gsl,
        # todo
        lr=config_classifier.training.lr,
        weight_decay=config_classifier.training.weight_decay,
        d_type=d_type,
        attr_to_classify=config_classifier.model.attr_to_classify,
        eval_attr_to_classify=config_classifier.model.eval_attr_to_classify,
        z_mean_std=z_mean_std,
    )
    # Trainer
    train_kwargs = {}
    results_dir = config_classifier.training.results_dir

    # logger setting
    logdir_path = osp.join(results_dir, 'checkpoints', 'ilab_classifier', f'{config_classifier.training.training_name}--checkpoints')
    tb_looger = pl_loggers.TensorBoardLogger(save_dir=logdir_path,
                                             name='Tensorboard_logs',
                                             version=config_classifier.training.training_name + \
                                                     time.strftime('%Y-%m-%d--%H_%M_%S')
                                             )
    tb_looger.log_hyperparams(OmegaConf.to_container(config_classifier, resolve=True))
    train_kwargs['logger'] = tb_looger

    # Setup callbacks
    checkpoint_path = os.path.join(logdir_path, 'Model_weights')
    if not osp.exists(checkpoint_path):
        makedirs(checkpoint_path)
    chkpt_callback = ModelCheckpoint(
        monitor='acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        dirpath=checkpoint_path,
        filename=f"model_classifier-{config_classifier.training.chkpt_prefix}" + "-{epoch:04d}-{global_step:09d}-{acc:.6f}",
        every_n_epochs=config_classifier.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )
    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config_classifier.training.epochs
    train_kwargs["log_every_n_steps"] = config_classifier.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    device = config_classifier.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs['accelerator'] = 'gpu'
        train_kwargs["devices"] = devs

        if len(devs) > 1:
            # Disable find_unused_parameters when using DDP training for performance reasons
            from pytorch_lightning.plugins import DDPPlugin
            train_kwargs["accelerator"] = "ddp"
            train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)

        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config_classifier.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config_classifier.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)

    # restore from checkpoint
    restore_path = config_classifier.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["ckpt_path"] = restore_path
        logger.info(f'Training from {restore_path}')
    elif osp.exists(osp.join(logdir_path, 'Model_weights', 'last.ckpt')):
        train_kwargs["ckpt_path"] = osp.join(logdir_path, 'Model_weights', 'last.ckpt')
        logger.info(f"Training from {osp.join(logdir_path, 'Model_weights', 'last.ckpt')}")
    else:
        train_kwargs["ckpt_path"] = None
        logger.info('A fresh start')

    trainer.fit(classifier, train_dataloaders=loader, ckpt_path=train_kwargs['ckpt_path'])


if __name__ == '__main__':
    train()
