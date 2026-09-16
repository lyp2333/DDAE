from torchvision import transforms as T
from .dataset_util import *


class Celeba64Lmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=64,
                 as_tensor=True,
                 do_norm=True,
                 do_augment=True,
                 ):
        self.data = BaseLMDB(path, image_size, zfill=7, is_group_dataset=False)
        self.length = len(self.data)
        self.transforms = []
        if as_tensor:
            self.transforms.append(T.ToTensor())
        if do_augment:
            self.transforms.append(T.RandomHorizontalFlip())
        if do_norm:
            self.transforms.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transforms = T.Compose(self.transforms)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transforms(img)
        return img


if __name__ == '__main__':
    dataset = Celeba64Lmdb('/home/lyp/Data/celeba/celeba.lmdb')
    print(len(dataset))
    print(dataset[2].shape)
