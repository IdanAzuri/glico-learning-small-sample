#!/usr/bin/env python3
import csv
import sys
import time

import easyargs
import matplotlib
import numpy as np
import os
import random
import torch
import torchvision
from torch import optim, nn
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import utils as vutils
matplotlib.use('Agg')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils import AverageMeter, get_loader_with_idx, maybe_cuda, get_cifar_param, get_cub_param, get_classifier, \
	ToTensor
from cGAN.model import Generator, Discriminator, load_model
from glico_model.cifar100 import get_cifar100, manual_seed
from cub2011 import Cub2011


def accuracy(model, test_data, batch_size, topk=(1, 5), aug_param=None):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	model.eval()
	if aug_param is None:
		aug_param = {'std': None, 'mean': None, 'rand_crop': 32, 'image_size': 32}
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	with torch.no_grad():
		test_loader = get_loader_with_idx(test_data, batch_size=batch_size, **aug_param, num_workers=8,
		                                  shuffle=False, eval=True)
		for i, batch in enumerate(test_loader):
			imgs = maybe_cuda(batch[1])
			targets = maybe_cuda(batch[2])
			output = maybe_cuda(model(imgs))
			maxk = max(topk)
			batch_size = targets.size(0)
			# target = validate_loader_consistency(batch_size, idx, target, test_data)
			_, pred = output.topk(maxk, 1, True, True)
			pred = pred.t()
			correct = pred.eq(targets.view(1, -1).expand_as(pred))
			res = []
			for k in topk:
				correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
				res.append(correct_k.mul_(100.0 / batch_size))
			top1.update(res[0].item())
			top5.update(res[1].item())
	return top1.avg, top5.avg


def generic_train_classifier(generator, model, optimizer, train_labeled_dataset, num_epochs, batch_size=512,
                             fewshot=False,
                             augment=False, autoaugment=False, test_data=None, n_classes=100,
                             aug_param=None, aug_param_test=None, shot=None):
	model.train()
	best_acc = 0
	milestones = [60, 120, 160]
	num_epoch_repetition = 1
	if fewshot:
		num_epoch_repetition = fewshot_setting(shot, aug_param['image_size'])
		milestones = list(map(lambda x: num_epoch_repetition * x, milestones))
		num_epochs *= num_epoch_repetition
	train_data_loader = get_loader_with_idx(train_labeled_dataset, batch_size, num_workers=8,
	                                        augment=augment,
	                                        offset_idx=0, offset_label=0, sampler=None,
	                                        autoaugment=autoaugment, cutout=False, random_erase=False,
	                                        **aug_param)
	print(f" train_len:{len(train_labeled_dataset)}")
	correct = 0
	total = 0
	if aug_param['image_size'] == 256:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2 * num_epoch_repetition, gamma=0.9)
	else:
		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

	for epoch in range(num_epochs):
		generator.eval()
		end_epoch_time = time.time()

		for i, (idx, inputs, targets) in enumerate(train_data_loader):
			inputs = inputs.cuda()
			optimizer.zero_grad()
			targets = targets.cuda()
			output = calc_output(generator, model, targets, inputs, aug_param)
			loss = criterion(output, targets)
			loss.backward()
			optimizer.step()
			_, predicted = torch.max(output.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).sum()
		print(
				f"Epoch [{epoch}/{num_epochs}] Acc:{100. * correct / total:2.2f}% loss:{loss.item():4.2f} lr:{scheduler.get_lr()[0]}  Time: {(time.time() - end_epoch_time) / 60:4.2f}m")
		scheduler.step()
		if epoch % 30 == 0 and test_data is not None:
			trans_acc_1, trans_acc_5 = accuracy(model, test_data, batch_size=batch_size, aug_param=aug_param_test)
			if trans_acc_1 > best_acc:
				best_acc = trans_acc_1
			print(f"\n=> Epoch[{epoch}] mid accuracy@1: {trans_acc_1} | accuracy@5: {trans_acc_5}\n")
	print(f"Best Acc 1 = {best_acc}")


def fewshot_setting(samples_per_class, im_size):
	print("=> fewshot")
	if im_size == 32:
		train_repeats = 500 // int(samples_per_class)
	else:
		train_repeats = 30 // int(samples_per_class)
	print(f"num_epoch_repetition: {train_repeats}")
	return train_repeats


def calc_output(generator_, model, targets, inputs, aug_param):
	normalize = transforms.Normalize(mean=aug_param['mean'], std=aug_param['std'])

	noise = torch.randn(len(targets), 100).cuda()
	if random.random() > 0.6:
		generated_img = generator_(noise, targets)
		# vutils.save_image(inputs.detach(), f'inputs.png',normalize=True)
		# vutils.save_image(generated_img.detach(), f'cGAN2.png',normalize=True)
		try:
			imgs = torch.stack([normalize((img)) for img in generated_img])
		except:
			imgs = torch.stack([normalize((img)) for img in generated_img.detach().cpu()])
		# vutils.save_image(imgs.data, f'cGAN_normalize.png',normalize=True)
	else:
		imgs = inputs
	output = model(imgs)
	return output


@easyargs
def run_eval_gan(epochs=200, d="wideresnet", fewshot=False, augment=False, shot=50,
                 unlabeled_shot=1, data="cifar", pretrained=False, seed=0):
	global aug_param, criterion, aug_param_test, batch_size
	manual_seed(seed)
	title = f"GAN_{d}"
	if fewshot:
		title = title + "_fewshot"
	dir = "eval"
	if not os.path.isdir(dir):
		os.mkdir(dir)
	data_dir = '../../data'
	cifar_dir_cs = '/cs/dataset/CIFAR/'
	if data == "cifar":
		classes = 100
		lr = 0.1
		batch_size = 128
		WD = 5e-4
		aug_param = aug_param_test = get_cifar_param()
		if not fewshot:
			train_labeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
			train_unlabeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
			test_data = train_unlabeled_dataset
		else:
			print("=> Fewshot")
			# train_labeled_dataset, train_unlabeled_dataset = get_cifar100_small(cifar_dir_small, shot)
			train_labeled_dataset, train_unlabeled_dataset, _, test_data = get_cifar100(cifar_dir_cs, n_labeled=shot,
			                                                                            n_unlabled=unlabeled_shot)

	if data == "cub":
		classes = 200
		batch_size = 16
		lr = 0.001
		WD = 1e-5
		aug_param = aug_param_test = get_cub_param()
		split_file = None
		if fewshot:
			samples_per_class = int(shot)
			split_file = 'train_test_split_{}.txt'.format(samples_per_class)
		train_labeled_dataset = Cub2011(root=f"../../data/{data}", train=True, split_file=split_file)
		train_unlabeled_dataset = Cub2011(root=f"../../data/{data}", train=False, split_file=split_file)
		test_data = Cub2011(root=f"../../data/{data}", train=False)

	train_data_size = len(train_labeled_dataset)
	print(f"train_labeled_dataset size:{train_data_size}")
	print(f"test_data size:{len(test_data)}")
	print(f"transductive data size:{len(train_unlabeled_dataset)}")

	set_seed(seed)
	generator = Generator()
	discriminator = Discriminator()
	classifier = get_classifier(classes, d, pretrained)
	epoch = 180
	generator, _ = load_model(epoch, generator, discriminator)
	generator=generator.cuda()
	optimizer = optim.SGD(classifier.parameters(), lr, momentum=0.9, weight_decay=WD, nesterov=True)
	print("=> Train new classifier")
	criterion = nn.CrossEntropyLoss().cuda()
	num_gpus = torch.cuda.device_count()
	if num_gpus > 1:
		print(f"=> Using {num_gpus} GPUs")
		classifier = nn.DataParallel(classifier).cuda()
		# cudnn.benchmark = True
	else:
		classifier = classifier.cuda()

	print(' => Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))
	print(f"=> {d}  Training model")
	print(f"=> Training Epochs = {str(epochs)}")
	print(f"=> Initial Learning Rate = {str(lr)}")
	generic_train_classifier(generator, classifier, optimizer, train_labeled_dataset,
	                         batch_size=batch_size, num_epochs=epochs,
	                         augment=augment,
	                         fewshot=fewshot, test_data=test_data,
	                         n_classes=classes, aug_param=aug_param,
	                         aug_param_test=aug_param_test, shot=shot
	                         )

	print("=> Done training classifier")

	valid_acc_1, valid_acc_5 = accuracy(classifier, train_labeled_dataset, batch_size=batch_size,
	                                    aug_param=aug_param_test)
	test_acc_1, test_acc_5 = accuracy(classifier, test_data, batch_size=batch_size, aug_param=aug_param_test)

	print('train_acc accuracy@1', valid_acc_1)
	print('train_acc accuracy@5', valid_acc_5)
	print('test accuracy@1', test_acc_1)
	print('test accuracy@5', test_acc_5)



def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)


if __name__ == '__main__':
	run_eval_gan()
	print("Done!")
