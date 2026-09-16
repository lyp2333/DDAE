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
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from main.models.gsl_net import Generator_fc
from main.util import configure_device, get_dataset
from lightning_fabric.utilities.seed import seed_everything
from tqdm import tqdm


def get_labels_index(labels_root):
    with open(labels_root, 'r') as f:
        lines = f.readlines()
        idx = [int(i) for i in range(len(lines))]
        labels = [lines[i].split(' ')[0] for i in idx]
        id_labels = []
        back_labels = []
        pose_labels = []
        for label in labels:
            splited_label = label.split('-')
            id_labels.append(splited_label[0])
            back_labels.append(splited_label[2])
            pose_labels.append(splited_label[3] + '-' + splited_label[4])
    id_classes, back_classes, pose_classes = map(sorted, (set(id_labels), set(back_labels), set(pose_labels)))

    logger.info(('id', len(id_classes), id_classes))
    logger.info(('background', len(back_classes), back_classes))
    logger.info(('pose', len(pose_classes), pose_classes))

    id_classes2index = dict((classes, idx) for idx, classes in enumerate(id_classes))
    back_classes2index = dict((classes, idx) for idx, classes in enumerate(back_classes))
    pose_classes2index = dict((classes, idx) for idx, classes in enumerate(pose_classes))

    id_index = [id_classes2index[key] for key in id_labels]
    back_index = [back_classes2index[key] for key in back_labels]
    pose_index = [pose_classes2index[key] for key in pose_labels]

    return id_index, back_index, pose_index


def get_eval_index(
    labels_root,
    label_list: list,
    attribute='id',
    dtype='ilab',
    num_to_eval: int = 1000,
):
    if dtype == 'ilab':
        attr2index = {
            'id': 0,
            'back': 1,
            'pose': 2,
        }
    else:
        raise NotImplementedError()
    all_index = get_labels_index(labels_root)
    index = all_index[attr2index[attribute]]

    res = []
    for i, label_i in enumerate(index):
        if i >= num_to_eval:
            break
        if label_i in label_list:
            res.append(i)
    # res = defaultdict(int)
    # for i, label_i in enumerate(index):
    #     if i >= num_to_eval:
    #         break
    #     if label_i in label_list:
    #         res[label_i] += 1
    return res


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="../../configs", config_name="default_conf")
def gsl_visualization(config):
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
    dim_slice = []
    if 'ilab' in d_type:
        ATTRIBUTE = ['id', 'back', 'pose']
        dim_slice = [slice(0, 60),
                     slice(60, 80),
                     slice(80, 100)]
    elif 'fonts' in d_type:
        for i in range(0, 100, 20):
            dim_slice.append(slice(i, i + 20))
    else:
        raise NotImplementedError()

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

    # Do not change model to eval model
    batch_size = min(config_GSL.evaluation.batch_size, len(dataset))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    # Generate sample from GSL-NET
    id_clabel, back_clabel, pose_clabel = get_labels_index(config_GSL.data.labels_root)
    z_dis_all_attr = [[] for _ in range(len(dim_slice))]
    visualization_save_path = config_GSL.evaluation.visualization_save_path
    if not os.path.exists(visualization_save_path):
        os.makedirs(visualization_save_path)
    savepath_2d = os.path.join(visualization_save_path, f'tsne-2d.pkl')
    num_samples = config_GSL.evaluation.num_tsne_samples
    total = 0
    if not os.path.exists(savepath_2d):
        with tqdm(total=num_samples * 5) as pbar:
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    batch = batch.to(dev)
                    z_dis = model.get_latent_code(batch)
                    for i, val in enumerate(dim_slice):
                        z_dis_all_attr[i].append(z_dis[:, dim_slice[i]])
                    pbar.update(batch_size)
                    total += batch_size
                    if total >= num_samples * 5:
                        break

        results2d = []
        for i, attr_z in enumerate(z_dis_all_attr):
            if i == 2:
                break
            elif i == 0:
                matrix2d = TSNE(n_components=2, n_iter=4000, verbose=True, perplexity=30).fit_transform(torch.cat(attr_z, dim=0)[:num_samples].cpu())
            else:
                matrix2d = TSNE(n_components=2, n_iter=4000, verbose=True, perplexity=30).fit_transform(torch.cat(attr_z, dim=0)[:num_samples * 5].cpu())
            results2d.append(matrix2d)
        torch.save(results2d, savepath_2d)
    else:
        results2d = torch.load(savepath_2d, map_location='cpu')
    id_label_index = get_eval_index(
        labels_root=config_GSL.data.labels_root,
        label_list=[1, 3, 4, 6, 8, 9],
        attribute='id',
        dtype='ilab',
        num_to_eval=num_samples,
    )

    back_label_index = get_eval_index(
        labels_root=config_GSL.data.labels_root,
        label_list=[52, 0, 7, 100, 3, 31, 86, 101, 13, 27, 88, 62, 82, 108, 105, 80, 95, 1],
        attribute='back',
        dtype='ilab',
        num_to_eval=num_samples * 5,
    )

    # visualization
    # 2D
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')

    plt.scatter(results2d[0][id_label_index, 0], results2d[0][id_label_index, 1], c=[id_clabel[i] for i in id_label_index], cmap='tab20')
    plt.savefig(os.path.join(visualization_save_path, 'tsne-id.png'), dpi=800)
    plt.clf()
    plt.scatter(results2d[1][back_label_index, 0], results2d[1][back_label_index, 1], c=[back_clabel[i] for i in back_label_index], cmap='tab20')
    plt.savefig(os.path.join(visualization_save_path, 'tsne-back.png'), dpi=800)
    plt.clf()

    print(results2d[0][id_label_index, 0].shape)
    print(results2d[1][back_label_index, 0].shape)
    logger.info('2d completed')


if __name__ == '__main__':
    gsl_visualization()
