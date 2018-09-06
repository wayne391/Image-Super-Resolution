import os
import torch
import pytorch_ssim
import pandas as pd
import numpy as np
from PIL import Image
from math import log10
from utils import traverse_dir, rgb2y_channel
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


class EvalDataset(Dataset):
    """
    Class for evaluation datasets

    Attributes
    ----------
    zip_list : list of tuples
        tuple[0]: file name of list 1
        tuple[1]: file name of list 2
    """
    def __init__(self, zip_list):
        super(EvalDataset, self).__init__()
        self.zip_list = zip_list

    def __getitem__(self, idx):
        tmp = self.zip_list[idx]
        img1_pil = preproc(tmp[0])
        img2_pil = preproc(tmp[1])

        img1_tensor = ToTensor()(img1_pil)
        img2_tensor = ToTensor()(img2_pil)
        return tmp[0], tmp[1], img1_tensor, img2_tensor

    def __len__(self):
        return len(self.zip_list)


def calc_ssim(img1, img2):
    """
    Caculate structural Similarity Index
    """
    ssim = pytorch_ssim.ssim(img1, img2).item()
    return ssim


def calc_psnr(img1, img2, max_=1):
    """
    Caculate PSNR
    """
    mse = ((img1 - img2)**2).data.mean()
    psnr = 10 * log10(max_**2 / mse)
    return psnr


def calc_score(img1, img2, crop=0):
    """
    Caculate ssim and PSNR
    """
    with torch.no_grad():
        img1 = Variable(img1)
        img2 = Variable(img2)
        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()
    psnr = calc_psnr(img1, img2)
    ssim = calc_ssim(img1, img2)
    return psnr, ssim


def preproc(filename):
    """
    Convert to Ycbcr and return PIL with only Y channel
    """
    img_pil = Image.open(filename).convert('RGB')
    img_Y = rgb2y_channel(np.array(img_pil))
    img_Y_pil = Image.fromarray(np.uint8(img_Y), "L")
    return img_Y_pil


def eval_image_file(img_file1, img_file2):
    """
    Example Function.
    Evaluate two images. The inputs are filenames
    """
    img1_pil = preproc(img_file1)
    img2_pil = preproc(img_file2)
    img1_tensor = torch.unsqueeze(ToTensor()(img1_pil), 0)
    img2_tensor = torch.unsqueeze(ToTensor()(img2_pil), 0)
    return calc_score(img1_tensor, img2_tensor)


def eval_filelist(filelist1, filelist2, verbose=True):
    """
    evaluate file list
    """
    if len(filelist1) != len(filelist2):
        raise ValueError('the amount should be equal')

    score_info = {'psnr': [], 'ssim': []}
    zip_list = list(zip(filelist1, filelist2))
    eval_set = EvalDataset(zip_list)
    eval_loader = DataLoader(dataset=eval_set, num_workers=4, batch_size=1, shuffle=False)
    for idx, (fn1, fn2, img1, img2) in enumerate(eval_loader):
        psnr, ssim = calc_score(img1, img2)
        if verbose:
            print('[%d] - %s  |  %s' %
                  (idx, os.path.basename(fn1[0]), os.path.basename(fn2[0])))
            print('   psnr: %.6f | ssim: %.6f\n' % (psnr, ssim))
        score_info['psnr'].append(psnr)
        score_info['ssim'].append(ssim)
    return score_info


def eval_benchmark(dir_benchmark, scale=2, verbose=False, dir_='./'):
    """
    Running Benchmarks. Srouce: https://github.com/jbhuang0604/SelfExSR
    [Warning] The results are slightly lower than original implematation in matlab
    """
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    # define benchmark and method
    benchmarks = ['Set5', 'Set14', 'BSD100', 'Urban100']
    methods = ['bicubic', 'nearest', 'glasner', 'Kim', 'SelfExSR', 'SRCNN']

    # scale
    scale_str = 'image_SRF_' + str(scale)

    # start eval
    print('{:=^40}'.format(' runnung benchmarks '))
    for benchmark in benchmarks:
        print('[%s]' % benchmark)
        root = os.path.join(dir_benchmark, benchmark, scale_str)
        method_result = {'psnr': [], 'ssim': []}

        for method in methods:
            print('  - %s' % method)

            # get list of files
            filelist_hr = traverse_dir(root, extension=('png'), str_='HR', is_sort=True)
            filelist_sr = [f.replace('HR', method) for f in filelist_hr]

            # eval
            score_info = eval_filelist(filelist_hr, filelist_sr, verbose=verbose)

            method_result['psnr'].append(np.mean(score_info['psnr']))
            method_result['ssim'].append(np.mean(score_info['ssim']))

        # save result
        data_frame = pd.DataFrame(method_result, index=methods)
        data_frame.to_csv(os.path.join(dir_, benchmark+'.csv'), index_label='method')
    print('{:=^40}'.format(' Done!!! '))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    eval_benchmark('datasets/benchmark', dir_='datasets/benchmark')
