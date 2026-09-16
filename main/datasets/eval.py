import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=None,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
    ):
        super().__init__()
        if exts is None:
            exts = ['jpg', 'png']
        self.folder = folder
        self.group_num = len(os.listdir(self.folder))
        self.image_size = image_size
        self.paths = [
            p.absolute() for i in range(self.group_num) for ext in exts
            for p in Path(f'{folder}').glob(f'*{i + 1}/[a-z].{ext}')
        ]

        self.path = sorted(self.paths)
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class DoubleImageDataset(Dataset):
    def __init__(
        self,
        folder1,
        folder2,
        image_size,
        exts=None,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
    ):
        super().__init__()
        if exts is None:
            exts = ['jpg']
        self.folder1 = folder1
        self.folder2 = folder2
        self.image_size = image_size
        self.folder1_paths = [
            p.absolute() for ext in exts for p in Path(f'{folder1}').glob(f'*.{ext}')
        ]
        self.folder2_paths = [
            p.absolute() for ext in exts for p in Path(f'{folder2}').glob(f'*.{ext}')
        ]

        self.folder1_paths = sorted(self.folder1_paths)
        self.folder2_paths = sorted(self.folder2_paths)
        assert len(self.folder1_paths) == len(self.folder2_paths)
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.folder1_paths)

    def __getitem__(self, index):
        path1 = self.folder1_paths[index]
        path2 = self.folder2_paths[index]
        img1 = Image.open(path1)
        img2 = Image.open(path2)

        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return {
            'img0': img1,
            'img1': img2,
        }


class RecombineDataset(Dataset):
    def __init__(self,
                 folder,
                 image_size,
                 dir_name='ilab',
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 ):
        self.dirname = dir_name
        self.groups_path = [os.path.join(folder, self.dirname, path) for path in os.listdir(Path(folder).joinpath(self.dirname))]
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.groups_path)

    def __getitem__(self, index):
        group_path = self.groups_path[index]
        A_img_path = os.path.join(group_path, 'id.jpg')
        B_img_path = os.path.join(group_path, 'background.jpg')
        D_img_path = os.path.join(group_path, 'pose.jpg')

        A_img = self.transform(Image.open(A_img_path).convert('RGB'))
        B_img = self.transform(Image.open(B_img_path).convert('RGB'))
        D_img = self.transform(Image.open(D_img_path).convert('RGB'))

        return torch.stack([A_img, B_img, D_img], dim=0)
