import os
import lmdb
import numpy as np
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torchvision.transforms import functional as Ftrans

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


def group_path(dir):
    num_dirs = 0
    path = []
    for root, dirs, files in os.walk(dir):  # 遍历统计
        for name in dirs:
            num_dirs += 1
            path.append(os.path.join(root, name))
            path.sort()
    return num_dirs, path


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)


class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5, is_group_dataset=False, **kwargs):

        self.is_group_dataset = is_group_dataset
        self.original_resolution = original_resolution
        self.zfill = zfill
        if "group_size" in kwargs.keys():
            self.group_size = kwargs['group_size']
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not self.is_group_dataset:
            with self.env.begin(write=False) as txn:
                key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
                    'utf-8')
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)
            return img
        else:
            with self.env.begin(write=False) as txn:
                key = f'group_{str(index).zfill(self.zfill)}'.encode('utf-8')
                group_bytes = txn.get(key)
            group_img: np.ndarray = np.frombuffer(group_bytes, dtype=np.uint8)
            group_img = group_img.copy().reshape(self.group_size, self.original_resolution, self.original_resolution, 3)
            return group_img
