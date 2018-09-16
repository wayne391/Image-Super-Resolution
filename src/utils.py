import os
import torch
import random
import numpy as np


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(args.save_dir, args.exp_name)

        # folder for saving
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # log
        log_file = os.path.join(self.save_dir, 'log.txt')
        if os.path.exists(log_file):
            self.logFile = open(log_file, 'a')
        else:
            self.logFile = open(log_file, 'w')

    def save_log(self, log):
        self.logFile.write(log + '\n')

    def save_model(self, model, name='model'):
        torch.save(
            model.state_dict(),
            os.path.join(self.save_dir, name+'_state.pt'))
        # torch.save(
        #     model,
        #     os.path.join(self.save_dir, name+'_obj.pt'))

    def load(self, model, name='model'):
        load_path = os.path.join(self.save_dir, name+'_state.pt')
        model.load_state_dict(torch.load(load_path))
        print(" [*] loaded from %s" % load_path)
        return model


def traverse_dir(
                root_dir,
                extension=('.jpg', 'png'),
                str_=None,
                is_pure=False,
                verbose=False,
                is_sort=False,
                is_ext=True):
    """
    Evaluate two images. The inputs are specified by file names
    """
    if verbose:
        print('[*] Scanning...')
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
    if verbose:
        print('Total: %d images' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list


ycbcr_para = np.array(
    [[65.481, 128.553, 24.966],
     [-37.797, -74.203, 112.0],
     [112.0, -93.786, -18.214]])


def rgb2ycbcr(npy):
    """
    Customized rgb to tcbcr function.
    """
    shape = npy.shape
    if len(shape) == 3:
        npy = npy.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(npy, ycbcr_para.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    return ycbcr.reshape(shape)


def rgb2y_channel(npy):
    """
    Convert RGB to Y channel
    """
    npy_ycbcr = rgb2ycbcr(npy)
    npy_y = npy_ycbcr[:, :, 0]
    return npy_y


def get_patch(img_lr, img_hr, patch_size, scale):
    (ih, iw, c) = img_lr.shape
    tp = patch_size
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    img_lr = img_lr[iy:iy + ip, ix:ix + ip, :]
    img_hr = img_hr[ty:ty + tp, tx:tx + tp, :]
    return img_lr, img_hr


# codes for data augmentation
PARA_LIST = [(0, 0), (0, 1), (0, 2), (0, 3),
             (1, 0), (1, 1), (1, 2), (1, 3)]

PARA_LIST_INV = [(0, 0), (0, 3), (0, 2), (0, 1),
                 (1, 0), (1, 1), (1, 2), (1, 3)]


def flip_vert(img):
    return img[::-1, :, :]


def flip_hori(img):
    return img[:, ::-1, :]


def rot(img, k=1):
    return np.rot90(img, k)


def augment_random(img1, img2):
    rand = random.randint(0, 7)
    p, k = PARA_LIST[rand]
    if p:
        img1 = flip_vert(img1)
        img2 = flip_vert(img2)
    return rot(img1, k), rot(img2, k)


'''
Self-ensemble (from EDSR):
First,

'''


def augment_all(img):
    proc_list = []
    for p, k in PARA_LIST:
        if p:
            proc_img = flip_vert(img)
        else:
            proc_img = img[:, :, :]
        proc_list.append(rot(proc_img, k))
    return proc_list


def self_ensemble(aug_list):
    proc_list = []
    sum_all = np.zeros_like(aug_list[0], dtype=float)
    for idx, (p, k) in enumerate(PARA_LIST_INV):
        img = aug_list[idx]
        if p:
            proc_img = flip_vert(img)
        else:
            proc_img = img
        result = rot(proc_img, k)
        sum_all = np.add(sum_all, result)
        proc_list.append(result)
    return sum_all/8.0, proc_list
