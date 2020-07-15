#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from glico_model.cifar100 import get_cifar100
from models import vgg19_bn
from create_dataset_cifar100 import CompatibleDataset
from glico_model.utils import _load_lowshot_cifar, AverageMeter, get_classifier

# from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="wideresnet", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--shot', default=0, type=int,
                    help='shot smaple')
parser.add_argument('--pretrained', '-pre', action='store_true',
                    help='pretrained')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
	torch.manual_seed(args.seed)
# Data
print('==> Preparing data..')
if args.augment:
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
	])
else:
	transform_train = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),

	])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.507, 0.487, 0.441), (0.2023, 0.1994, 0.2010)),
])

# trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
#                             transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True, num_workers=8)
#

# train_data_imgs, train_lables = _load_lowshot_cifar(data_dir="../../data/cifar-100", split="train", num_shot=args.shot)
# train_data = CompatibleDataset(train_data_imgs, train_lables, transform=transform_train)
# train_data = datasets.CIFAR100(root='../../data/', train=True, download=False,
#                             transform=transform_train)
#
# testset = datasets.CIFAR100(root='../../data/', train=False, download=False,
#                             transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                          shuffle=False, num_workers=8)
# testset = _load_lowshot_cifar(data_dir="../../data/cifar-100", split="test",
#                                                           num_shot=args.shot)
cifar_dir_cs = '/cs/dataset/CIFAR/'
train_labeled_dataset, train_unlabeled_dataset, _, test_data = get_cifar100(cifar_dir_cs, n_labeled=args.shot,
                                                                            n_unlabled=10)
trainloader = torch.utils.data.DataLoader(train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(test_data, batch_size=100,
                                         shuffle=False, num_workers=8)
# Model
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
	                        + str(args.seed))
	net = checkpoint['net']
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch'] + 1
	rng_state = checkpoint['rng_state']
	torch.set_rng_state(rng_state)
else:
	print('==> Building model..')
	# net = vgg19_bn(num_classes=100)
	net = get_classifier(100, args.model, args.pretrained)

if not os.path.isdir('results'):
	os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net)
	print(torch.cuda.device_count())
	cudnn.benchmark = True
	print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay, nesterov=True)
milestones = [60, 120, 160]
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	reg_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()

		inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
		                                               args.alpha, use_cuda)
		inputs, targets_a, targets_b = map(Variable, (inputs,
		                                              targets_a, targets_b))
		outputs = net(inputs)
		loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
		train_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
		            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# progress_bar(batch_idx, len(trainloader),
		#              'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
		#              % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
		#                 100.*correct/total, correct, total))
		print('Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
		      % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
		         100. * correct / total, correct, total))
	return (train_loss / batch_idx, reg_loss / batch_idx, 100. * correct / total)


def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	topk = (1, 5)
	for batch_idx, (inputs, targets) in enumerate(testloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		outputs = net(inputs)

		maxk = max(topk)
		batch_size = targets.size(0)
		# target = validate_loader_consistency(batch_size, idx, target, test_data)
		_, pred = outputs.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(targets.view(1, -1).expand_as(pred))
		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		acc = res[0].item()
		top1.update(acc)
		top5.update(res[1].item())

		# progress_bar(batch_idx, len(testloader),
		#              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		#              % (test_loss/(batch_idx+1), 100.*correct/total,
		#                 correct, total))
		print('Loss: %.3f | Acc: %.3f' % (test_loss / (batch_idx + 1), acc))
	if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
		checkpoint(acc, epoch)
	if acc > best_acc:
		best_acc = acc
	# return (test_loss/batch_idx, 100. * correct / total)
	return (test_loss / batch_idx, top1.avg, top5.avg)


def checkpoint(acc, epoch):
	# Save checkpoint.
	print('Saving..')
	state = {
		'net': net,
		'acc': acc,
		'epoch': epoch,
		'rng_state': torch.get_rng_state()
	}
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')
	torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
	           + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
	"""decrease the learning rate at 100 and 150 epoch"""
	lr = args.lr
	if epoch >= 100:
		lr /= 10
	if epoch >= 150:
		lr /= 10
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if not os.path.exists(logname):
	with open(logname, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
		                    'test loss', 'test acc'])

scores_test_acc1_fewshot = dict()
scores_test_acc5_fewshot = dict()
for seed in [71, 132, 54]:
	torch.manual_seed(seed)
	for epoch in range(start_epoch, args.epoch):
		train_loss, reg_loss, train_acc = train(epoch)
		if epoch % 10 == 0:
			test_loss, test_acc1, test_acc5 = test(epoch)
		scheduler.step()
		# adjust_learning_rate(optimizer, epoch)
		with open(logname, 'a') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
			                    test_acc1])

	print('accuracy@1', test_acc1)
	print('accuracy@5', test_acc5)

	scores_test_acc1_fewshot[seed] = test_acc1
	scores_test_acc5_fewshot[seed] = test_acc5

	w = csv.writer(open(f"mixup_shot_{args.shot}_acc1.csv", "w+"))
	for key, val in scores_test_acc1_fewshot.items():
		w.writerow([key, val])
	w = csv.writer(open(f"mixup_shot_{args.shot}_acc5.csv", "w+"))
	for key, val in scores_test_acc5_fewshot.items():
		w.writerow([key, val])
