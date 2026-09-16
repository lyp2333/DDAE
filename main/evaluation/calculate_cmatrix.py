import torch
import os
import sys
import logging
import copy
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

sys.path.append("/home/lyp/Code/DiffuseGAE")
from collections import OrderedDict
from torch.utils.data import DataLoader
from main.models.diffusion import Encoder
from main.models.autoencoders.GAE import GAE_PL
from main.models.gsl_net import Generator_fc
from main.models.classifier import ClsModel
from main.util import configure_device, get_dataset, parse_str
from lightning_fabric.utilities.seed import seed_everything
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="../configs", config_name="default_conf")
def calculate_cmatrix(config):
    # Eval classifier use the same config file as classifier training
    config_joint = config.dataset.joint
    config_classifier = config.dataset.classifier
    attr_to_classify = 'back'

    dev = torch.device('cuda:0')

    # Dataset & dataloader
    if config_classifier.model.z_norm:
        z = torch.load(config_classifier.data.z_root)
        z_mean_std = (z["mean"], z["std"])
    else:
        z_mean_std = None
    d_type = config_classifier.data.name
    dataset = get_dataset(
        d_type,
        config_classifier.data.root,
        config_classifier.data.image_size,
        config_classifier.data.norm,
        False,
    )
    N = len(dataset)
    batch_size = config_classifier.training.batch_size
    batch_size = min(N, batch_size)
    eval_num = 40000
    n_iter = eval_num % batch_size + 1
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config_classifier.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )


    # Model
    # GAE setting
    gae_pl = GAE_PL.load_from_checkpoint(config_classifier.model.GAE_ckpt_path, strict=False)
    gae_pl.eval()

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

    # classifier
    weights_path = os.path.join(config_classifier.training.training_name, 'Model_weights', 'last.ckpt')
    classifier_pl = ClsModel.load_from_checkpoint(weights_path, strict=True)
    classifier = classifier_pl.classifier.to(dev).eval()

    if 'ilab' in d_type:
        dim_slices = {'id': slice(0, 60),
                      'back': slice(60, 80),
                      'pose': slice(80, 100)}

    else:
        raise NotImplementedError()
    train_mode = classifier_pl.train_mode
    acc_list = []
    with tqdm(n_iter) as pbar:
        for i, batch in enumerate(loader):
            imgs = batch['img']
            batch_size, *_ = imgs.shape
            labels = batch[f'{attr_to_classify}_label']
            with torch.no_grad():
                if train_mode == "encoder_out":
                    z_original = encoder(imgs)
                    z_for_classify = z_original
                elif train_mode == "gae_out":
                    z_original = encoder(imgs)
                    z_in_gae = (z_original - z_mean_std[0]) / z_mean_std[1] if z_mean_std is not None else z_original
                    z_for_classify = gae_pl.get_latent_code(z_in_gae)
                elif train_mode == "gsl_out":
                    z_for_classify = gsl.get_latent_code(imgs)
                else:
                    raise NotImplementedError()
            pred = classifier(z_for_classify) if train_mode == 'encoder_out' else classifier(z_for_classify[:, dim_slices[attr_to_classify]])
            train_acc = torch.sum(torch.where(torch.argmax(pred, dim=-1) == labels, 1, 0)) / batch_size
            pbar.update(1)
