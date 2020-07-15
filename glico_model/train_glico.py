import argparse
import os

import torch
import torchvision

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from glico_model.cifar10 import get_cifar10
from glico_model.cifar100 import get_cifar100, manual_seed
from cub2011 import Cub2011
from glico_model import nag_trainer
from glico_model.utils import NAGParams, OptParams
import socket

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.register('type', 'bool', (lambda x: x.lower() in ("yes", "true", "t", "1")))
parser.add_argument('--rn', type=str, help='file name')
parser.add_argument('--decay', type=int, default=70, help='lr decay every num images')
parser.add_argument('--gamma', type=float, default=0.5, help='factor of cross entropy loss in nag train')
parser.add_argument('--epoch', type=int, default=202, help='total_epoch')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--factor', type=float, default=0.7, help='factor between G_lr and Z_lr')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pixel', action='store_true', help='determine if discriminator is pixel sapce or latent space')
# parser.add_argument('--classifier', action='store_true', help='determine if classifier')
parser.add_argument('--resume', action='store_true', help='determine if resume')
parser.add_argument('--z_init', type=str, default='rndm', help='z init method', choices=['cube', 'resnet', 'rndm'])
parser.add_argument('--shot', type=int, default=0, help='num of small sample')
parser.add_argument('--unlabeled_shot', type=int, default=0, help='num of small sample')
parser.add_argument('--batch_size', type=int, default=128, help='num of small sample')
parser.add_argument('--dim', type=int, default=512, help='num of small sample')
parser.add_argument('--tr', action='store_true', help='is transductive learning')
parser.add_argument('--fewshot', action='store_true', help='is fewshot')
parser.add_argument('--noise_proj', action='store_true', help='noise proj')
parser.add_argument('--add_noise', action='store_true', help='addative noise')
parser.add_argument('--d', type=str, choices=['vgg', 'wideresnet', 'none', 'conv', 'resnet'], default='none',
                    help='classifier name')
parser.add_argument('--data', type=str, choices=['cifar', 'cub', 'stl', 'cifar-10'], default='cifar', help='dataset')
parser.add_argument('--loss', type=str, choices=['ce', 'cosine', 'ce_smooth'], default='ce', help='loss')
args = parser.parse_args()
# training parameters

print(args)
manual_seed(args.seed)

is_classifier = args.d != "none"

data_dir = '../../data'
cifar_dir_cs = '/cs/dataset/CIFAR/'
dataset = args.data
if dataset == 'cifar':
    data_name = 'cifar-100'
if dataset == 'cifar-10':
    data_name = 'cifar-10'
if dataset == 'cub':
    data_name = 'cub'
if dataset == 'stl':
    data_name = 'stl'
test_dataset = []
if data_name == 'cifar-100':
    classes = 100
    batch_size = min(args.batch_size, 512)
    if args.fewshot:
        print("=> Fewshot")
        # train_labeled_dataset, train_unlabeled_dataset = get_cifar100_small(cifar_dir_small, args.shot)
        train_labeled_dataset, train_unlabeled_dataset, _, test_dataset = get_cifar100(cifar_dir_cs,
                                                                                       n_labeled=args.shot,
                                                                                       n_unlabled=args.unlabeled_shot)
    else:
        train_labeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
        train_unlabeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
if data_name == 'cifar-10':
    classes = 10
    batch_size = min(args.batch_size, 512)
    if args.fewshot:
        print("=> Fewshot")
        # train_labeled_dataset, train_unlabeled_dataset = get_cifar100_small(cifar_dir_small, args.shot)
        train_labeled_dataset, train_unlabeled_dataset, _, test_dataset = get_cifar10(cifar_dir_cs,
                                                                                      n_labeled=args.shot,
                                                                                      n_unlabled=args.unlabeled_shot)
    else:
        train_labeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
        train_unlabeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
if dataset == 'cub':
    split_file = None
    if args.fewshot:
        samples_per_class = int(args.shot)
        split_file = 'train_test_split_{}.txt'.format(samples_per_class)
    # train_repeats = 30 // samples_per_class
    classes = 200
    batch_size = min(args.batch_size, 32)
    train_labeled_dataset = Cub2011(root=f"/cs/labs/daphna/idan.azuri/data/cub", train=True, split_file=split_file)
    train_unlabeled_dataset = Cub2011(root=f"/cs/labs/daphna/idan.azuri/data/cub", train=False, split_file=split_file)
    test_dataset = []
if dataset == "stl":
    print("STL-10")
    classes = 10
    batch_size = min(args.batch_size, 32)
    train_labeled_dataset = torchvision.datasets.STL10(root=f"../../data/{dataset}", split='train', download=True)
    train_unlabeled_dataset = torchvision.datasets.STL10(root=f"../../data/{dataset}", split='unlabeled', download=True)
    train_unlabeled_dataset = torch.utils.data.Subset(train_unlabeled_dataset, list(range(0, 10000)))
# test_data = torchvision.datasets.STL10(root=f"../../data/{args.data}", split='test', download=True)

nag_params = NAGParams(nz=args.dim, force_l2=False, is_pixel=args.pixel, z_init=args.z_init,
                       is_classifier=is_classifier, disc_net=args.d, loss=args.loss, data_name=data_name,
                       noise_proj=args.noise_proj, shot=args.shot)
nag_opt_params = OptParams(lr=args.lr, factor=args.factor, batch_size=batch_size, epochs=args.epoch,
                           decay_epochs=args.decay, decay_rate=0.5, gamma=args.gamma)


def get_run_name_from_args(args):
    # args_str = ""
    # for arg_name, arg_val in vars(args).items():
    # 	if arg_val is not None and arg_val:
    # 		if type(arg_val) == bool:
    # 			args_str = f"{args_str}_{arg_name}"
    # 		else:
    # 			args_str = f"{args_str}_{arg_val}"
    # print(args_str)
    arg_str = ""
    if args.pixel:
        arg_str = f"{arg_str}_pixel"
    elif is_classifier:
        arg_str = f"{arg_str}_latent"
    if is_classifier:
        arg_str = f"{arg_str}_classifier_{args.d}"
    if args.tr:
        arg_str = f"{arg_str}_tr"
    if args.fewshot:
        arg_str = f"{arg_str}_fs_{args.shot}"
    arg_str = f"{arg_str}_{args.loss}"
    if args.noise_proj:
        arg_str = f"{arg_str}_noise_proj"
    elif args.add_noise:
        arg_str = f"{arg_str}_add_noise"

    return f"{data_name}_{args.rn}{arg_str}_{args.z_init}_{nag_opt_params.decay_epochs}_{nag_opt_params.lr}_{nag_opt_params.factor}_{nag_opt_params.decay_rate}_gamma{nag_opt_params.gamma}"


rn = get_run_name_from_args(args)

if args.tr:

    nt = nag_trainer.NAGTrainer([train_labeled_dataset, train_unlabeled_dataset, test_dataset], nag_params, rn,
                                resume=args.resume,
                                num_classes=classes)
else:
    nt = nag_trainer.NAGTrainer([train_labeled_dataset, [], []], nag_params, rn, resume=args.resume,
                                num_classes=classes)

nt.train_test_nag(nag_opt_params)
