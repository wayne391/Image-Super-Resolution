import os
import argparse
import trainer
from data import DIV2KDataSet
# from model.RDN import SRNet
from models.RCAN import SRNet

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

# ----- Global Setting -----
GPU = 1
EPOCH = 6000
P = 196
STEP = 1000  # for optimizer scheduler
nohup = '../exp'
EXP_DIR = '../debug'

# train
parser.add_argument('--gpu', type=int, default=GPU, help='gpu')
parser.add_argument('--epochs', type=int, default=EPOCH, help='number of epochs')
parser.add_argument('--is_finetuning', default=False, help='finetune the model')
parser.add_argument('--step_print_loss', type=int, default=100, help='stpes to print loss')
parser.add_argument('--step_save', type=int, default=800, help='stpes to print loss')

parser.add_argument('--num_threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--scheduler_step_size', type=float, default=STEP, help='period of learning rate decay')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='decay ratio')

# data
parser.add_argument('--data_dir', default='../datasets/DIV2K', help='dataset directory')
parser.add_argument('--benchmark_dir', default='../datasets/benchmark', help='benchmark directory')
parser.add_argument('--need_patch', default=True, help='get patch form image')
parser.add_argument('--patch_size', type=int, default=P,  help='patch size (P)')
parser.add_argument('--scale', type=int, default=2, help='scale')

# ----- model (RDN) -----
# C = 6
# G = 32
# D = 10
# F = 64
# EXP_NAME = '[%s]_GPU_%d---EPOCH_%d-P_%d---C_%d-G_%d-D_%d-F_%d' % (nohup, GPU, EPOCH, P, C, G, D, F)
# parser.add_argument('--num_dense', type=int, default=C, help='number of conv layer in RDB (C)')
# parser.add_argument('--growth_rate', type=int, default=G, help='growth rate of dense net (G)')
# parser.add_argument('--num_RDB', type=int, default=D, help='number of RDB block (D)')
# parser.add_argument('--num_feat', type=int, default=F,  help='number of conv feature maps (F)')
# parser.add_argument('--num_channel', type=int, default=3, help='number of color channels to use')

# ----- model (RCAN) -----
F = 64
NB = 12
NG = 6
EXP_NAME = '[%s]_GPU_%s---EPOCH_%d-P_%d---F_%d-NB_%d-NG_%d' % (nohup, str(GPU), EPOCH, P, F, NB, NG)
parser.add_argument('--num_feat', type=int, default=F,  help='number of conv feature maps (F)')
parser.add_argument('--num_channel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--n_resblocks', type=int, default=NB, help='number of residual blocks (NB)')
parser.add_argument('--n_resgroups', type=int, default=NG, help='number of residual groups (NG)')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')

# Saver
parser.add_argument('--save_dir', default=EXP_DIR, help='datasave directory')
parser.add_argument('--exp_name', default=EXP_NAME, help='save result')

args = parser.parse_args()


def main():
    # prepare datasets
    train_set = DIV2KDataSet(args, args.scale, args.data_dir, 'DIV2K_train_HR', 'DIV2K_train_LR_bicubic')
    test_set = DIV2KDataSet(args, args.scale, args.data_dir, 'DIV2K_valid_HR', 'DIV2K_valid_LR_bicubic')

    # append datasets
    # train_set_flickr2k = DIV2KDataSet(args, args.scale, '../datasets/Flickr2K',
    #                                   'Flickr2K_HR', 'Flickr2K_LR_bicubic')
    # train_set.append(train_set_flickr2k)
    print(train_set.__len__())

    # model
    my_model = SRNet(args)

    if isinstance(args.gpu, list) and len(args.gpu) > 1:
        import torch.nn as nn
        my_model = nn.DataParallel(my_model, args.gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    trainer.network_paras(my_model)

    # restor for testing/fine-tuning
    if args.is_finetuning:
        my_model = trainer.restore(args, my_model)

    # train
    trainer.train(args, my_model, train_set)

    # run benchmark
    trainer.run_benchmark(args, my_model, args.benchmark_dir)

    # test
    trainer.test(args, my_model, test_set)


if __name__ == '__main__':
    main()
