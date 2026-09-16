import torch
import lpips
import sys

sys.path.extend(['/home/lyp/Code/DiffuseGAE', '/home/lyp/Code/DiffuseGAE/third_party'])
import logging
import hydra
from main.trainer.metric import ssim, psnr
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from main.datasets.eval import DoubleImageDataset
from main.util import configure_device
from lightning_fabric.utilities.seed import seed_everything
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="../configs", config_name="default_conf.yaml")
def autoencoding_evaluate(config):
    logger.info(OmegaConf.to_yaml(config))
    config_joint = config.dataset.joint
    # Set seed
    seed_everything(config_joint.evaluation.seed)
    scores = {
        'lpips': [],
        'mse': [],
        'ssim': [],
        'psnr': [],
    }
    # Dataset
    # folder should be absolute path
    folder1 = config_joint.data.autoencoding_imgs_root0
    folder2 = config_joint.data.autoencoding_imgs_root1
    assert Path(folder1).is_absolute()
    assert Path(folder2).is_absolute()

    dev_ = configure_device(config_joint.evaluation.device)
    dev = dev_[1] if isinstance(dev_, tuple) else dev_

    dev = torch.device(f'cuda:{dev[0]}' if isinstance(dev, list) else dev)

    image_size = config_joint.data.image_size
    dataset = DoubleImageDataset(
        folder1=folder1,
        folder2=folder2,
        image_size=image_size,
        exts=['png', 'jpg'],
        do_augment=False,
    )
    batch_size = config_joint.evaluation.batch_size
    workers = config_joint.evaluation.workers
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    lpips_fn = lpips.LPIPS(net='alex').to(dev)
    with torch.no_grad():
        with tqdm(loader, total=len(dataset)) as pbar:
            for idx, batch in enumerate(loader):
                real_bsize = batch['img0'].shape[0]
                imgs0, imgs1 = batch['img0'].to(dev), batch['img1'].to(dev)
                scores['lpips'].append(lpips_fn(imgs0, imgs1).view(-1))
                norm_imgs, norm_pred_imgs = (imgs0 + 1) / 2, (imgs1 + 1) / 2
                scores['ssim'].append(
                    ssim(norm_imgs, norm_pred_imgs, size_average=False))
                scores['mse'].append(
                    (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
                scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))
                pbar.update(real_bsize)
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float().mean().item()
        logger.info(scores)
        with open('/home/lyp/Code/DiffuseGAE/autoencoding_evaluation.txt', 'a+') as f:
            for key in scores.keys():
                f.write(f'{key}: {scores[key]} \n')


if __name__ == '__main__':
    autoencoding_evaluate()
