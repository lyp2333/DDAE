import torch
import random
from torchvision import transforms as T
from .dataset_util import *


class FontLmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=128,
                 as_tensor=True,
                 do_norm=True,
                 do_augment=True,
                 ):
        self.data = BaseLMDB(path, image_size, zfill=8, is_group_dataset=False)
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


class FontsDataset(Dataset):
    '''
    Content / size / color(Font) / color(background) / style
    E.g. A / 64/ red / blue / arial
    C random sample
    AC same content; BC same size; DC same font_color; EC same back_color; FC same style
    '''

    def __init__(self, root, norm=True, transform=None, train=True):
        super(FontsDataset, self).__init__()
        self.train = train
        self.norm = norm
        self.root = root
        self.transform = transform
        # self.paths = make_dataset(self.root)
        if self.train:
            self.C_size = 52  # too much we fix it as the number of letters
            '''refer'''
            # color 10
            self.Colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
                           'cyan': (0, 255, 255),
                           'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30),
                           'silver': (192, 192, 192)}
            self.Colors = list(self.Colors.keys())
            # size 3
            self.Sizes = {'small': 80, 'medium': 100, 'large': 120}
            self.Sizes = list(self.Sizes.keys())
            # style nearly over 100
            for roots, dirs, files in os.walk(os.path.join(self.root, 'A', 'medium', 'red', 'orange')):
                cates = dirs
                break
            self.All_fonts = cates
            # letter 52
            self.Letters = [chr(x) for x in list(range(65, 91)) + list(range(97, 123))]
        else:  # test mode
            self.C_size, self.paths = group_path(self.root)  # size of center image C

    def findN(self, index):
        # random choose a C image
        C_letter = self.Letters[index]
        C_size = random.choice(self.Sizes)
        C_font_color = random.choice(self.Colors)
        resume_colors = self.Colors.copy()
        resume_colors.remove(C_font_color)
        C_back_color = random.choice(resume_colors)
        C_font = random.choice(self.All_fonts)
        C_img_name = C_letter + '_' + C_size + '_' + C_font_color + '_' + C_back_color + '_' + C_font + ".png"
        C_img_path = os.path.join(self.root, C_letter, C_size, C_font_color, C_back_color, C_font, C_img_name)
        ''' exclusive the C attribute avoid same with C'''
        temp_Letters = self.Letters.copy()  # avoid same size with C
        temp_Letters.remove(C_letter)
        temp_Size = self.Sizes.copy()  # avoid same size with C
        temp_Size.remove(C_size)
        temp_font_color = self.Colors.copy()  # avoid same font_color with C
        temp_font_color.remove(C_font_color)
        temp_back_colors = self.Colors.copy()  # avoid same back_color with C and avoid same color with font
        temp_back_colors.remove(C_back_color)
        temp_font = self.All_fonts.copy()  # avoid same font with C
        temp_font.remove(C_font)

        # A has same content
        '''SAME content'''
        A_letter = C_letter
        A_size = random.choice(temp_Size)
        A_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if A_font_color in resume_colors:
            resume_colors.remove(A_font_color)
        A_back_color = random.choice(resume_colors)
        A_font = random.choice(temp_font)
        A_img_name = A_letter + '_' + A_size + '_' + A_font_color + '_' + A_back_color + '_' + A_font + ".png"
        A_img_path = os.path.join(self.root, A_letter, A_size, A_font_color, A_back_color, A_font, A_img_name)

        # B has same size
        B_letter = random.choice(temp_Letters)
        '''SAME size'''
        B_size = C_size
        B_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if B_font_color in resume_colors:
            resume_colors.remove(B_font_color)
        B_back_color = random.choice(resume_colors)
        B_font = random.choice(temp_font)
        B_img_name = B_letter + '_' + B_size + '_' + B_font_color + '_' + B_back_color + '_' + B_font + ".png"
        B_img_path = os.path.join(self.root, B_letter, B_size, B_font_color, B_back_color, B_font, B_img_name)

        # D has same font_color
        D_letter = random.choice(temp_Letters)
        D_size = random.choice(temp_Size)
        '''SAME font_color'''
        D_font_color = C_font_color
        resume_colors = temp_back_colors.copy()
        if D_font_color in resume_colors:
            resume_colors.remove(D_font_color)
        D_back_color = random.choice(resume_colors)
        D_font = random.choice(temp_font)
        D_img_name = D_letter + '_' + D_size + '_' + D_font_color + '_' + D_back_color + '_' + D_font + ".png"
        D_img_path = os.path.join(self.root, D_letter, D_size, D_font_color, D_back_color, D_font, D_img_name)

        # E has same back_color
        E_letter = random.choice(temp_Letters)
        E_size = random.choice(temp_Size)
        resume_colors = temp_font_color.copy()
        resume_colors.remove(C_back_color)
        E_font_color = random.choice(resume_colors)
        '''SAME back_color'''
        E_back_color = C_back_color
        E_font = random.choice(temp_font)
        E_img_name = E_letter + '_' + E_size + '_' + E_font_color + '_' + E_back_color + '_' + E_font + ".png"
        E_img_path = os.path.join(self.root, E_letter, E_size, E_font_color, E_back_color, E_font, E_img_name)

        # F has same font
        F_letter = random.choice(temp_Letters)
        F_size = random.choice(temp_Size)
        F_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if F_font_color in resume_colors:
            resume_colors.remove(F_font_color)
        F_back_color = random.choice(resume_colors)
        '''SAME font'''
        F_font = C_font
        F_img_name = F_letter + '_' + F_size + '_' + F_font_color + '_' + F_back_color + '_' + F_font + ".png"
        F_img_path = os.path.join(self.root, F_letter, F_size, F_font_color, F_back_color, F_font, F_img_name)

        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path

    def findtest(self, index):
        '''
                    refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style
                    A2B3D4E5F1_combine_2N
        A: size provider
        B: font_color provider
        D: back_color provider
        E: font provider
        F: letter provider
        '''
        group_path = self.paths[index]
        A_img_path = os.path.join(group_path, 'size.png')
        B_img_path = os.path.join(group_path, 'font_color.png')
        D_img_path = os.path.join(group_path, 'back_color.png')
        E_img_path = os.path.join(group_path, 'font.png')
        F_img_path = os.path.join(group_path, 'letter.png')

        return A_img_path, B_img_path, D_img_path, E_img_path, F_img_path

    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        if self.train:
            A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path = self.findN(index)

            A_img = Image.open(C_img_path).convert('RGB')
            B_img = Image.open(A_img_path).convert('RGB')
            C_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')
            E_img = Image.open(E_img_path).convert('RGB')
            F_img = Image.open(F_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                C = self.transform(C_img)
                D = self.transform(D_img)
                E = self.transform(E_img)
                F = self.transform(F_img)
            res = {}
            for key, img in zip(('A', 'B', 'C', 'D', 'E', 'F'), (A, B, C, D, E, F)):
                res[key] = img
            return res
        else:  # test
            A_img_path, B_img_path, D_img_path, E_img_path, F_img_path = self.findtest(index)
            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')
            E_img = Image.open(E_img_path).convert('RGB')
            F_img = Image.open(F_img_path).convert('RGB')

            if self.transform is not None:
                A = self.transform(A_img)
                B = self.transform(B_img)
                D = self.transform(D_img)
                E = self.transform(E_img)
                F = self.transform(F_img)

            return {'A': A, 'B': B, 'D': D, 'E': E, 'F': F}

    def __len__(self):
        return self.C_size


class GroupFontsLmdb(Dataset):
    def __init__(self,
                 path,
                 group_size: int = 6,
                 image_size: int = 128,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
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


if __name__ == '__main__':
    dataset = FontLmdb('/home/lyp/Data/Fonts/fonts_128.lmdb')
    print(dataset[1].shape)

