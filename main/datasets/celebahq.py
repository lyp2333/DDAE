from torchvision import transforms as T
from .dataset_util import *


class Celebahq256Lmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=256,
                 as_tensor=True,
                 do_norm=True,
                 do_augment=True,
                 ):
        self.data = BaseLMDB(path, image_size, zfill=5, is_group_dataset=False)
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
    dataset = Celebahq256Lmdb('/home/lyp/Data/celebahq/celebahq256.lmdb')
    print(dataset[1].shape)
