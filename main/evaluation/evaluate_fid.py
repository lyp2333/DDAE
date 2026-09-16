import logging
import random
import os
import shutil
import time
import multiprocessing as mp
import torch
import glob
from tqdm import trange
from pytorch_fid import fid_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_fid_score(path: tuple,
                  batch_size=256,
                  device=torch.device('cuda:0'),
                  dims=2048,
                  ):
    print('calculating fid score')
    score_fid = fid_score.calculate_fid_given_paths(
        path,
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    return score_fid


def imgs_copy(original_paths: list,
              target_path: str,
              num_samples: int):
    for i in range(num_samples):
        shutil.copy(original_paths[i], target_path + '/' + f'{i + 1}'.zfill(5) + '.png')


if __name__ == '__main__':
    fid_path = '/home/lyp/Code/DiffuseGAE/Recon/fid.txt'
    if os.path.exists(fid_path):
        os.remove(fid_path)
    for i in trange(100):
        start = time.perf_counter()
        random.seed(i)
        original_path = "/home/lyp/Code/DiffuseGAE/Recon/original"
        recon_path = "/home/lyp/Code/DiffuseGAE/Recon/recon"
        num_samples = 10000
        target_original_path = "/home/lyp/Code/DiffuseGAE/Recon/test_original"
        target_recon_path = "/home/lyp/Code/DiffuseGAE/Recon/test_recon"
        if not os.path.exists(target_original_path):
            os.makedirs(target_original_path)
        if not os.path.exists(target_recon_path):
            os.makedirs(target_recon_path)
        original_path1 = glob.glob(original_path + '/*.png')
        original_path2 = glob.glob(recon_path + '/*.png')

        # path1 = glob.glob(target_original_path + '/*.png')
        # path2 = glob.glob(target_recon_path + '/*.png')
        # if len(path1) != num_samples or len(path2) != num_samples:
        #     print("before: ", len(glob.glob(target_recon_path + '/*.png')))
        #     shutil.rmtree(target_original_path)
        #     shutil.rmtree(target_recon_path)
        #     os.makedirs(target_original_path)
        #     os.makedirs(target_recon_path)
        #
        #     for i in trange(num_samples):
        #         shutil.copy(original_path1[i], target_original_path + '/' + f'{i + 1}'.zfill(5) + '.png')
        #         shutil.copy(original_path2[i], target_recon_path + '/' + f'{i + 1}'.zfill(5) + '.png')
        #     print("after: ", len(glob.glob(target_recon_path + '/*.png')))
        # todo delete start
        shutil.rmtree(target_original_path)
        shutil.rmtree(target_recon_path)
        os.makedirs(target_original_path)
        os.makedirs(target_recon_path)

        path_index = random.sample(range(50000), num_samples)
        path1 = [original_path1[i] for i in path_index]
        path2 = [original_path2[i] for i in path_index]
        # todo delete end

        # todo
        process_group = [mp.Process(target=imgs_copy, args=(path1, target_original_path, num_samples)),
                         mp.Process(target=imgs_copy, args=(path2, target_recon_path, num_samples))]
        for p in process_group:
            p.start()
            p.join()
        fid = get_fid_score((target_original_path, target_recon_path), dims=2048)
        end = time.perf_counter()
        with open(f'/home/lyp/Code/DiffuseGAE/Recon/fid.txt', "a+") as f:
            f.write(f'seed{i}: {fid} time_consumed: {end - start}s\n')

