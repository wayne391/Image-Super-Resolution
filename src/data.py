import os
import numpy as np
from PIL import Image
from utils import traverse_dir, get_patch, augment_random

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


# meta class for dataset classes
class MetaSet(Dataset):
    """
    Meta Class for various src

    Usage:
    Must define four vars in the __init__

    Depending on 'file_list_hr':
    traing stage:     list of img
    validation stage: list of img
    testing stage:    list of None

    Note that no matter what the stage is, the length of 'file_list_hr'
    should be the same as 'file_list_lr'
    """
    def __init__(self):
        # basic
        self.args = None
        self.file_list = None
        self.file_list_hr = None
        self.file_list_lr = None
        self.num_train = None

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_hr = self.file_list_hr[idx]
        file_lr = self.file_list_lr[idx]

        pil_hr = Image.open(file_hr).convert('RGB')
        pil_lr = Image.open(file_lr).convert('RGB')

        if self.args.need_patch:
            img_hr = np.array(pil_hr)
            img_lr = np.array(pil_lr)
            img_lr_patch, img_hr_patch = get_patch(img_lr, img_hr, self.args.patch_size, self.scale)
            img_lr_patch, img_hr_patch = augment_random(img_lr_patch, img_hr_patch)
            pil_hr = Image.fromarray(img_hr_patch, "RGB")
            pil_lr = Image.fromarray(img_lr_patch, "RGB")

        hr_tensor = ToTensor()(pil_hr)
        lr_tensor = ToTensor()(pil_lr)
        return filename, lr_tensor, hr_tensor

    def __len__(self):
        return self.num_train

    def append(self, new_set):
        self.file_list += new_set.file_list
        self.file_list_hr += new_set.file_list_hr
        self.file_list_lr += new_set.file_list_lr
        self.num_train += new_set.num_train


class BenchmarkSet(MetaSet):
    def __init__(self, args, dir_benchmark):
        # basic
        self.args = args
        self.scale = args.scale
        self.dir_benchmark = dir_benchmark

        # list arrangement
        self.file_list_hr = traverse_dir(dir_benchmark, extension=('png'), str_='HR', is_sort=True)
        self.file_list_lr = [f.replace('HR', 'LR') for f in self.file_list_hr]
        self.file_list = [f.replace('HR', 'new')[22:] for f in self.file_list_hr]

        # num of train
        self.num_train = len(self.file_list)


class DIV2KDataSet(MetaSet):
    def __init__(self, args, scale, data_dir, dir_hr, dir_lr):
        # basic
        self.args = args
        self.scale = scale

        # src dir
        self.dirHR = os.path.join(data_dir, dir_hr)
        self.dirLR = os.path.join(data_dir, dir_lr, 'X'+str(self.scale))
        self.file_list = traverse_dir(self.dirHR, extension=('png'), is_pure=True, is_ext=False)

        # re-arrange filename
        self.file_list_hr = [os.path.join(self.dirHR, f + '.png') for f in self.file_list]
        self.file_list_lr = [os.path.join(self.dirLR, f + 'x'+str(self.scale)+'.png') for f in self.file_list]
        self.file_list = [f + '.png' for f in self.file_list]  # for saving

        # num of train
        self.num_train = len(self.file_list)
