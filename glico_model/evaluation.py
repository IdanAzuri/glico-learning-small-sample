#!/usr/bin/env python3
import csv
import glob

import time

import easyargs
import matplotlib
import os
import random
from torch import optim
from torch.backends import cudnn
from torch.optim import lr_scheduler


matplotlib.use('Agg')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import glico_model
from glico_model.cifar10 import get_cifar10
from glico_model.cifar100 import get_cifar100, manual_seed
from cub2011 import Cub2011
from glico_model.interpolate import slerp_torch
from glico_model.utils import *
from glico_model.model import _netZ, _netG,  DCGAN_G, DCGAN_G_small
from logger import Logger

# dim = 512
code_size = 100


class LinearSVM(nn.Module):
	"""Support Vector Machine"""

	def __init__(self, num_class, dim):
		super(LinearSVM, self).__init__()
		self.fc = nn.Linear(dim, num_class)

	# self.sgi = nn.Sigmoid()

	def forward(self, x):
		h = self.fc(x)
		return h  # , torch.argmax(h,1)


# normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])  # CIFAR100
#
# transform_list = normalize


def accuracy(model, test_data, batch_size=128, topk=(1, 5), aug_param=None):
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


def generic_train_classifier(model, optimizer, train_labeled_dataset, num_epochs, is_inter, criterion, offset_idx=0,
                             offset_label=0, batch_size=128, fewshot=False,
                             augment=False, autoaugment=False, test_data=None, loss_method="ce", n_classes=100,
                             aug_param=None, aug_param_test=None, shot=None, cutout=False, random_erase=False,
                             is_lerp=False):
	model = model.cuda()
	model.train()
	sampler = None
	best_acc = 0
	milestones = [60, 120, 160]
	num_epoch_repetition = 1
	if fewshot:
		num_epoch_repetition = fewshot_setting(shot, aug_param['image_size'])
		milestones = list(map(lambda x: num_epoch_repetition * x, milestones))
		num_epochs *= num_epoch_repetition
	train_data_loader = get_loader_with_idx(train_labeled_dataset, batch_size,
	                                        augment=augment, shuffle=True,
	                                        offset_idx=offset_idx, offset_label=offset_label, sampler=sampler,
	                                        autoaugment=autoaugment, **aug_param, cutout=cutout,
	                                        random_erase=random_erase)
	print(f" train_len:{len(train_labeled_dataset)}")
	correct = 0
	total = 0
	if aug_param['image_size'] == 256:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2 * num_epoch_repetition, gamma=0.9)
	else:
		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

	for epoch in range(num_epochs):
		# netG.eval() # issue with batchnorm
		netZ.eval()
		model.train()
		end_epoch_time = time.time()

		for i, (idx, inputs, _) in enumerate(train_data_loader):
			inputs = inputs.cuda()
			optimizer.zero_grad()
			targets = validate_loader_consistency(netZ, idx)
			output = calc_output(idx, is_inter, model, targets, inputs, aug_param, is_lerp=is_lerp)
			targets = torch.tensor(targets).long().cuda()
			if loss_method == "cosine":
				y_target = torch.ones(len(idx)).cuda()
				onehot_labels = one_hot(targets, n_classes)
				loss = criterion(output, onehot_labels.cuda(), y_target)
			elif loss_method == "ce_smooth":  # cross entrop smoothing
				criterion = LabelSmoothingLoss(n_classes, smoothing=0.1)
				loss = criterion(output, onehot_labels.long())
			# targets=smooth_one_hot(targets,n_classes,smoothing=0.1).long()
			# loss = criterion(output, targets)
			else:
				loss = criterion(output, targets)
			loss.backward()
			optimizer.step()
			_, predicted = torch.max(output.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).sum()
		# if i % 100 ==0 :
		# sys.stdout.write('\r')
		# sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
		#                  % (epoch, num_epochs, i + 1,
		#                     (len(train_labeled_dataset) // batch_size) + 1, loss.item(), 100. * correct / total))
		# sys.stdout.flush()
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
	return max(train_repeats,1)


def calc_output(idx, is_inter, model, targets, inputs, aug_param, is_lerp):
	# Zs_real_numpy = netZ.emb.weight.data.cpu().numpy()
	normalize = transforms.Normalize(mean=aug_param['mean'], std=aug_param['std'])
	transform_train_np = transforms.Compose([
			# RandomPadandCrop(32),
			# RandomFlip(),
			ToTensor(),
			normalize,
			])
	Zs_real = netZ.emb.weight.data
	if is_inter:
		# z = Zs_real_numpy[idx]
		z = Zs_real[idx]
		# z = z.detach().cpu().numpy()
		# z_knn_dist, z_knn = neigh.kneighbors(z, return_distance=True)
		# rnd_int = random.randint(1, 3)
		# z_knn = z_knn[:, 1:3]  # ignore self dist
		z_knn_idx = np.array([random.sample(netZ.label2idx[x], 1) for x in targets])

		# z_knn_idx = z_knn_idx[:, 0]# * 0.7 + z_knn[:, 1] * 0.3

		ratios = list(np.linspace(0.1, 0.4, 4))
		rnd_ratio = random.sample(ratios, 1)[0]
		# z = torch.from_numpy(z).float().cuda()
		z = z.float().cuda()
		z_knn = Zs_real[z_knn_idx[:, 0]]  # * 0.9 + Zs_real[z_knn_idx[:, 1]] * 0.1

		# eps=maybe_cuda(torch.FloatTensor(len(idx), dim).normal_(0, 0.01))
		if is_lerp:
			inter_z = torch.lerp(z, z_knn.float(), rnd_ratio)
		else:  # default
			inter_z = slerp_torch(rnd_ratio, z.unsqueeze(0), z_knn.cuda().float().unsqueeze(0))
		# inter_z_slerp = interpolate_points(z, z_knn.float(), 10, print_mode=False, slerp=True)
		# inter_z = interpolate_points(z,z_knn.float(),10,print_mode=False,slerp=False)
		# targets = torch.tensor(targets).long().cuda()
		# targets = targets.repeat(10)

		code = get_code(idx)
		if random.random() > 0.5:
			generated_img = netG(inter_z.squeeze().cuda(), code)

			try:
				imgs = torch.stack([normalize((img)) for img in generated_img.detach().cpu()])
			except:
				imgs = torch.stack([normalize((img)) for img in generated_img])

		else:
			imgs = inputs  # the input is augmented
		output = model(imgs.cuda()).squeeze()  # .cpu().numpy()


	else:
		code = get_code(idx)
		if random.random() > 0.5:
			generated_img = netG(Zs_real[idx].cuda(), code)
			imgs = torch.stack([normalize((img)) for img in generated_img])
		else:
			imgs = inputs
		output = model(imgs.cuda())
	return output


def get_code(idx):
	code = torch.cuda.FloatTensor(len(idx), code_size).normal_(0, 0.15)
	# normed_code = code.norm(2, 1).detach().unsqueeze(1).expand_as(code)
	# code = code.div(normed_code)
	return code


@easyargs
def run_eval_(is_inter=False, debug=False, keyword="", epochs=200, d="", fewshot=False, augment=False, shot=50,
              unlabeled_shot=50,
              autoaugment=False,
              loss_method="ce", data="cifar", pretrained=False, dim=512, seed=0, cutout=False, random_erase=False,
              lerp=False):
	global netG, netZ, Zs_real, neigh, aug_param, criterion, aug_param_test
	manual_seed(seed)
	title = f"{d}_{keyword}"
	if fewshot:
		title = title + "_fewshot"

	if is_inter:
		title = title + "_inter"
	dir = "eval"
	if not os.path.isdir(dir):
		os.mkdir(dir)
	logger = Logger(os.path.join(f'{dir}/', f'{title}_log.txt'), title=title)
	logger.set_names(["valid_acc@1", "valid_acc@5", "test_acc@1", "test_acc@5"])
	PATH = "/myglico_model/glico_model"
	data_dir = '../../data'
	cifar_dir_cs = '/cs/dataset/CIFAR/'
	if data == "cifar":
		classes = 100
		lr = 0.1
		batch_size = 128
		WD = 5e-4
		aug_param = aug_param_test = get_cifar100_param()
		if not fewshot:
			train_labeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
			train_unlabeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
			test_data = train_unlabeled_dataset
		else:
			print("=> Fewshot")
			# train_labeled_dataset, train_unlabeled_dataset = get_cifar100_small(cifar_dir_small, shot)
			train_labeled_dataset, train_unlabeled_dataset, _, test_data = get_cifar100(cifar_dir_cs, n_labeled=shot,
			                                                                            n_unlabled=unlabeled_shot)
	if data == "cifar-10":
		classes = 10
		lr = 0.1
		batch_size = 128
		WD = 5e-4
		aug_param = aug_param_test = get_cifar10_param()
		if not fewshot:
			train_labeled_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
			train_unlabeled_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
			test_data = train_unlabeled_dataset
		else:
			print("=> Fewshot")
			train_labeled_dataset, train_unlabeled_dataset, _, test_data = get_cifar10(cifar_dir_cs, n_labeled=shot,
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
	if data == "stl":
		print("STL-10")
		classes = 10
		WD = 4e-4
		batch_size = 32
		lr = 2e-3
		aug_param = get_train_n_unlabled_stl_param()
		aug_param_test = get_test_stl_param()
		train_labeled_dataset = torchvision.datasets.STL10(root=f"../../data/{data}", split='train', download=True)
		train_unlabeled_dataset = test_data = torchvision.datasets.STL10(root=f"../../data/{data}", split='unlabeled',
		                                                                 download=True)
	train_data_size = len(train_labeled_dataset)
	print(f"train_labeled_dataset size:{train_data_size}")
	print(f"test_data size:{len(test_data)}")
	print(f"transductive data size:{len(train_unlabeled_dataset)}")


	# netZ.set_label2idx()
	noise_projection = keyword.__contains__("proj")
	print(f" noise_projection={noise_projection}")
	# netG = _netG(dim, aug_param['rand_crop'], 3, noise_projection)
	if data == 'cub':
		netG = DCGAN_G(dim, aug_param['image_size'], 3, noise_projection, noise_projection)
		print(f"G: {dim}, {aug_param['image_size']}, 3, {noise_projection},{noise_projection}")
	elif data == 'stl':
		netG = DCGAN_G_small(dim, aug_param['image_size'], 3, noise_projection, noise_projection)
	elif 'cifar' in data:
		netG = _netG(dim, aug_param['image_size'], 3, noise_projection, noise_projection)
	paths = list()
	print(f"{PATH}")
	dirs = [d for d in glob.glob(PATH)]
	print(dirs)
	for dir in dirs:
		for f in glob.iglob(f"{dir}/runs/{keyword}*log.txt"):
			fname = f.split("/")[-1]
			tmp = fname.split("_")
			name = '_'.join(tmp[:-1])
			# if is_model_classifier:
			# 	if "classifier" in name or "cnn" in name:
			paths.append(name)
	# else:
	# 	paths.append(name)
	scores_test_acc1_fewshot = dict()
	scores_test_acc5_fewshot = dict()
	set_seed(seed)
	print(f"=> Total runs: {len(paths)}\n{paths}")
	for rn in paths:
		classifier = get_classifier(classes, d, pretrained)
		if "tr_" in rn:
			print("=> Transductive mode")
			netZ = _netZ(dim, train_data_size + len(train_unlabeled_dataset) + len(test_data), classes, None)
		# netZ = _netZ(dim, train_data_size +len(test_data), classes, None)
		else:
			print("=> No Transductive")
			netZ = _netZ(dim, train_data_size, classes, None)
		# 	netZ = _netZ(dim, train_data_size +len(test_data)+ len(train_unlabeled_dataset), classes, None)
		try:
			print(f"=> Loading model from {rn}")
			_, netG = load_saved_model(f'runs/nets_{rn}/netG_nag', netG)
			epoch, netZ = load_saved_model(f'runs/nets_{rn}/netZ_nag', netZ)
			netZ = netZ.cuda()
			netG = netG.cuda()
			print(f"=> Embedding size = {len(netZ.emb.weight)}")
			print(' => Total params: %.2fM' % (sum(p.numel() for p in netZ.parameters()) / 1000000.0))
			print(' => Total params: %.2fM' % (sum(p.numel() for p in netG.parameters()) / 1000000.0))
			if epoch > 0:
				print(f"=> Loaded successfully! epoch:{epoch}")
			else:
				print("=> No checkpoint to resume")
		except Exception as e:
			print(f"=> Failed resume job!\n {e}")
		Zs_real = netZ.emb.weight.data.detach().cpu().numpy()
		optimizer = optim.SGD(classifier.parameters(), lr, momentum=0.9, weight_decay=WD, nesterov=True)
		print("=> Train new classifier")
		if loss_method == "cosine":
			criterion = nn.CosineEmbeddingLoss().cuda()
		else:
			criterion = nn.CrossEntropyLoss().cuda()
		num_gpus = torch.cuda.device_count()
		if num_gpus > 1:
			print(f"=> Using {num_gpus} GPUs")
			classifier = nn.DataParallel(classifier).cuda()
			cudnn.benchmark = True
		else:
			classifier = maybe_cuda(classifier)

		print(' => Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))
		print(f"=> {d}  Training model")
		print(f"=> Training Epochs = {str(epochs)}")
		print(f"=> Initial Learning Rate = {str(lr)}")
		generic_train_classifier(classifier, optimizer, train_labeled_dataset, criterion=criterion,
		                         batch_size=batch_size,
		                         is_inter=is_inter, num_epochs=epochs, augment=augment,
		                         fewshot=fewshot, autoaugment=autoaugment, test_data=test_data,
		                         loss_method=loss_method, n_classes=classes, aug_param=aug_param,
		                         aug_param_test=aug_param_test, shot=shot, cutout=cutout, random_erase=random_erase,
		                         is_lerp=lerp)

		print("=> Done training classifier")

		valid_acc_1, valid_acc_5 = accuracy(classifier, train_labeled_dataset, batch_size=batch_size,
		                                    aug_param=aug_param_test)
		test_acc_1, test_acc_5 = accuracy(classifier, test_data, batch_size=batch_size, aug_param=aug_param_test)

		print('train_acc accuracy@1', valid_acc_1)
		print('train_acc accuracy@5', valid_acc_5)
		print('test accuracy@1', test_acc_1)
		print('test accuracy@5', test_acc_5)
		scores_test_acc1_fewshot[seed] = test_acc_1
		scores_test_acc5_fewshot[seed] = test_acc_5
		logger.append([valid_acc_1, valid_acc_5, test_acc_1, test_acc_5])
		w = csv.writer(open(f"eval_{rn}_shot_{shot}_acc1.csv", "w+"))
		for key, val in scores_test_acc1_fewshot.items():
			w.writerow([rn, key, val])
		w = csv.writer(open(f"eval_{rn}_shot_{shot}_acc5.csv", "w+"))
		for key, val in scores_test_acc5_fewshot.items():
			w.writerow([rn, key, val])

	logger.close()


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)


if __name__ == '__main__':
	run_eval_()
	print("Done!")
