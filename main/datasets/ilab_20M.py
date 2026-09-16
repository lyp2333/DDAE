import torch
import random
from torchvision import transforms as T
from .dataset_util import *


class Ilab128Lmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=128,
                 as_tensor=True,
                 do_norm=True,
                 do_augment=True,
                 ):
        self.data = BaseLMDB(path, image_size, zfill=6, is_group_dataset=False)
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


class Ilab128Lmdb_with_labels(Dataset):
    def __init__(self,
                 path,
                 image_size=128,
                 as_tensor=True,
                 do_norm=True,
                 do_augment=True,
                 ):
        self.data = BaseLMDB(path, image_size, zfill=6, is_group_dataset=False)
        self.label_root = '/home/lyp/Data/ilab/ilab_128.lmdb/ilab_128.txt'
        self.length = len(self.data)
        self.id_index, self.back_index, self.pose_index = map(torch.tensor, self.get_labels_index())
        self.transforms = []
        if as_tensor:
            self.transforms.append(T.ToTensor())
        if do_augment:
            self.transforms.append(T.RandomHorizontalFlip())
        if do_norm:
            self.transforms.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transforms = T.Compose(self.transforms)

    def get_labels_index(self):
        with open(self.label_root, 'r') as f:
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

        self.id_classes2index = dict((classes, idx) for idx, classes in enumerate(id_classes))
        self.back_classes2index = dict((classes, idx) for idx, classes in enumerate(back_classes))
        self.pose_classes2index = dict((classes, idx) for idx, classes in enumerate(pose_classes))

        id_index = [self.id_classes2index[key] for key in id_labels]
        back_index = [self.back_classes2index[key] for key in back_labels]
        pose_index = [self.pose_classes2index[key] for key in pose_labels]

        return id_index, back_index, pose_index

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transforms(img)
        return {
            "img": img,
            "id_label": self.id_index[index],
            "back_label": self.back_index[index],
            "pose_label": self.pose_index[index]
        }


class Ilab256Lmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=256,
                 as_tensor=True,
                 do_norm=True,
                 do_augment=True,
                 ):
        self.data = BaseLMDB(path, image_size, zfill=6, is_group_dataset=False)
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


class ILabDataset(Dataset):
    def __init__(self, root, norm=True, transform=None, train=True):
        super(ILabDataset, self).__init__()
        self.train = train
        self.norm = norm
        self.root = root
        self.transform = transform
        if self.train:
            self.paths = make_dataset(self.root)
            self.C_size = len(self.paths)  # size of center image C
        else:  # test mode
            self.C_size, self.paths = group_path(self.root)  # size of center image C

    def findABD(self, index):
        '''
        refer paper Fig.3
        C: x
        A: same pose as C
        B: same identity as C
        D: same background as C
        '''
        FOUNDED = False
        while not FOUNDED:
            FOUNDED = True
            C_img_path = self.paths[index % self.C_size]

            C_pose_root = C_img_path.split('/')[-2]
            C_pose = C_pose_root.replace('_', '-')
            C_category = C_img_path.split('/')[-1].split('-')[0]
            C_identity = C_img_path.split('/')[-1].split('-')[1]
            C_back = C_img_path.split('/')[-1].split('-')[2]

            # B has same identity as C
            B_root = self.root.replace('train_img_c00_10class', 'vae_identity_new')
            B_category = C_category
            B_identity = C_identity
            B_img_root = os.path.join(B_root, B_category, B_identity)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)
            if not C_pose in B_img_name and not C_back in B_img_name:
                B_img_path = os.path.join(B_img_root, B_img_name)
            else:
                # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 1000
                continue  # break the BREAK_ALL

            # A has same pose as C
            A_img_root = os.path.join(self.root, C_pose_root)
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            if not C_identity in A_img_name and not C_back in A_img_name:
                A_img_path = os.path.join(A_img_root, A_img_name)
            else:
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL

            # D has same back
            # back-cate-pose
            D_root = self.root.replace('train_img_c00_10class', 'vae_back_new')
            D_back = C_back
            # D has same back
            D_img_root_back = os.path.join(D_root, D_back)
            # D must have different identity and diff pose with C
            '''cate '''
            for roots, dirs, files in os.walk(D_img_root_back):
                cates = dirs
                break
            cates.remove(C_category)
            if len(cates) <= 0:  # no other category to choose
                # print('The D image can not have different cate with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL
            selected_D_cate = random.choice(cates)
            D_img_root_cate = os.path.join(D_img_root_back, selected_D_cate)
            '''pose '''
            for roots, dirs, files in os.walk(D_img_root_cate):
                poses = dirs
                break
            poses.remove(C_pose_root)

            if len(poses) <= 0:  # no other category to choose
                # print('The D image can not have different pose with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 2 if index < self.C_size - 20 else index - 200
                continue  # break the BREAK_ALL
            selected_D_pose = random.choice(poses)
            D_img_root = os.path.join(D_img_root_cate, selected_D_pose)
            D_files = os.listdir(D_img_root)
            D_image_index = random.randint(0, len(D_files) - 1)
            D_img_path = os.path.join(D_img_root, D_files[D_image_index])

        return A_img_path, B_img_path, C_img_path, D_img_path

    def findtest(self, index):
        '''
        A: id provider
        B: pose provider
        D: background provider
        '''
        group_path = self.paths[index]
        A_img_path = os.path.join(group_path, 'id.jpg')
        B_img_path = os.path.join(group_path, 'pose.jpg')
        D_img_path = os.path.join(group_path, 'background.jpg')
        return A_img_path, B_img_path, D_img_path

    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        if self.train:
            A_img_path, B_img_path, C_img_path, D_img_path = self.findABD(index)

            A_img = Image.open(C_img_path).convert('RGB')
            B_img = Image.open(A_img_path).convert('RGB')
            C_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                C = self.transform(C_img)
                D = self.transform(D_img)

            res = {}
            for key, img in zip(('A', 'B', 'C', 'D'), (A, B, C, D)):
                res[key] = img
            return res
        else:  # test
            A_img_path, B_img_path, D_img_path = self.findtest(index)
            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                D = self.transform(D_img)

            return {'A': A, 'B': B, 'D': D}

    def __len__(self):
        return self.C_size


class GroupIlabLmdb(Dataset):
    def __init__(self,
                 path,
                 group_size=4,
                 image_size=128,
                 as_tensor=True,
                 do_augment=True,
                 do_normalize=True,
                 ):
        self.group_size = group_size
        self.data = BaseLMDB(path, image_size, zfill=6, group_size=group_size, is_group_dataset=True)
        self.length = len(self.data)
        self.transform = []
        if as_tensor:
            self.transform.append(T.ToTensor())
        if do_augment:
            self.transform.append(T.RandomHorizontalFlip())
        if do_normalize:
            self.transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transform = T.Compose(self.transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        group_img = self.data[index].copy()
        tmp_group_img = []
        if self.transform:
            for img_index in range(self.group_size):
                tmp_group_img.append(self.transform(group_img[img_index, :, :, :]))
        group_img = torch.stack(tmp_group_img)
        return group_img


class GroupZIlab(Dataset):
    def __init__(self,
                 path,
                 group_size=4,
                 dataset_size=-1,
                 do_normalize=True,
                 ):
        assert dataset_size != 0, "dataset_size can not be 0"
        self.group_size = group_size
        self.data = torch.load(path, map_location="cpu")
        self.dataset_size = len(self.data) if dataset_size == -1 else dataset_size
        if do_normalize:
            # group_z
            self.z_mean = self.data['mean']
            self.z_std = self.data['std']
            self.z_data = (self.data['data'] - self.z_mean) / self.z_std
        else:
            self.z_data = self.data['data']

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self.z_data[index]


if __name__ == '__main__':
    # dataset = ILabDataset(root="/home/lyp/Data/ilab/ilab_256/train_img_c00_10class",
    #                       norm=False,
    #                       )
    # res = 0
    # print(len(dataset))
    # for root, dir_names, files in os.walk("/home/lyp/Data/ilab/ilab_256/"):
    #
    #     if 'train_img_c00_10class' in root:
    #         break
    #     for file in files:
    #         if is_image_file(file):
    #             res += 1
    # print(res)
    # print(dataset.paths[0])
    # dataset = GroupZIlab(
    #     path='/home/lyp/Code/DiffuseGAE/checkpoints/ilab_128-diffae-form1--checkpoints/latent_code/group_latent_code-ilab_128_group_z.pth',
    #     do_normalize=True,
    # )
    # dataset = Ilab128Lmdb_with_labels(path="/home/lyp/Data/ilab/ilab_128.lmdb",
    #                                   do_augment=False, )
    # print(dataset[0]['back_label'])
    dataset = Ilab128Lmdb(
        path="/home/lyp/Data/ilab/ilab_128.lmdb",
    )
    print(dataset[2].shape)
