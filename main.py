import os
import model
import argparse
import trainer
from data import DIV2KDataSet

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

# train
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--epochs', type=int, default=800, help='number of epochs')
parser.add_argument('--is_finetuning', default=False, help='finetune the model')
parser.add_argument('--step_print_loss', type=int, default=100, help='stpes to print loss')

parser.add_argument('--num_threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# saver
parser.add_argument('--save_dir', default='exp', help='datasave directory')
parser.add_argument('--exp_name', default='try_2dataset_800', help='save result')

# data
parser.add_argument('--data_dir', default='datasets/DIV2K', help='dataset directory')
parser.add_argument('--benchmark_dir', default='datasets/benchmark', help='benchmark directory')
parser.add_argument('--need_patch', default=True, help='get patch form image')
parser.add_argument('--patch_size', type=int, default=144,  help='patch size')
parser.add_argument('--scale', type=int, default=2, help='scale')

# model
parser.add_argument('--nDenselayer', type=int, default=4, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=64, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')

args = parser.parse_args()


def main():
    # prepare datasets
    train_set = DIV2KDataSet(args, args.scale, args.data_dir, 'DIV2K_train_HR', 'DIV2K_train_LR_bicubic')
    test_set = DIV2KDataSet(args, args.scale, args.data_dir, 'DIV2K_valid_HR', 'DIV2K_valid_LR_bicubic')
    train_set_flickr2k = DIV2KDataSet(args, args.scale, 'datasets/Flickr2K',
                                      'Flickr2K_HR', 'Flickr2K_LR_bicubic')
    train_set.append(train_set_flickr2k)
    print(train_set.__len__())

    # model
    my_model = ver2.RDN(args)

    # restor for testing/fine-tuning
    # my_model = trainer.restore(args, my_model)

    # train
    trainer.train(args, my_model, train_set)

    # test
    trainer.test(args, my_model, test_set)

    # run benchmark
    trainer.run_benchmark(args, my_model, args.benchmark_dir)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    main()
