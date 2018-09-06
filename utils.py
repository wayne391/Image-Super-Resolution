import os
import torch
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
        print(" [*] load from %s" % load_path)
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
