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
from glico_model.cifar100 import get_cifar100, manual_seed
from cub2011 import Cub2011
from glico_model.interpolate import slerp_torch, save_inter_imgs, save_inter_imgs_in_npz, save_dataset_in_npz
from glico_model.utils import *
from glico_model.model import _netZ, _netG, Classifier, DCGAN_G, DCGAN_G_small
from logger import Logger

# dim = 512
code_size = 100



def get_code(idx):
	code = torch.cuda.FloatTensor(len(idx), code_size).normal_(0, 0.15)
	# normed_code = code.norm(2, 1).detach().unsqueeze(1).expand_as(code)
	# code = code.div(normed_code)
	return code


@easyargs
def run_(is_inter=False, debug=False, keyword="", net="", fewshot=False, augment=False, shot=50,
              unlabeled_shot=50,
              autoaugment=False, data="cifar", pretrained=False, dim=512, seed=0, cutout=False, random_erase=False,):
	global netG, netZ, Zs_real, neigh, aug_param, criterion, aug_param_test
	manual_seed(seed)
	FULL_PATH = keyword
	data_dir = '../../data'
	cifar_dir_cs = '/cs/dataset/CIFAR/'
	if data == "cifar":
		classes = 100
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

	if data == "cub":
		classes = 200
		aug_param = aug_param_test = get_cub_param()
		split_file = None
		if fewshot:
			samples_per_class = int(shot)
			split_file = 'train_test_split_{}.txt'.format(samples_per_class)
		train_labeled_dataset = Cub2011(root=f"../../data/{data}", train=True, split_file=split_file)
		train_unlabeled_dataset = Cub2011(root=f"../../data/{data}", train=False, split_file=split_file)
		test_data = Cub2011(root=f"../../data/{data}", train=False)

	noise_projection = keyword.__contains__("proj")
	print(f" noise_projection={noise_projection}")
	# netG = _netG(dim, aug_param['rand_crop'], 3, noise_projection)
	if data == 'cub':
		netG = DCGAN_G(dim, aug_param['image_size'], 3, noise_projection, noise_projection)
	elif data == 'cifar':
		netG = _netG(dim, aug_param['image_size'], 3, noise_projection, noise_projection)
	data_to_save = test_data
	if keyword != "":
		set_seed(seed)
		if "tr_" in FULL_PATH:
			netZ = _netZ(dim, len(train_labeled_dataset) + len(train_unlabeled_dataset) + len(test_data), classes, None)
			# netZ = _netZ(dim, len(train_labeled_dataset) +len(train_unlabeled_dataset), classes, None)
		else:
			netZ = _netZ(dim, len(train_labeled_dataset), classes, None)
			# netZ = _netZ(dim, train_data_size +len(test_data)+ len(train_unlabeled_dataset), classes, None)
		print(f"=> Loading model from {FULL_PATH}")
		_, netG = load_saved_model(f'{FULL_PATH}/netG_nag', netG)
		epoch, netZ = load_saved_model(f'{FULL_PATH}/netZ_nag', netZ)
		# _, netG = load_saved_model(f'{PATH}/runs2/nets_{rn}/netG_nag', netG)
		# epoch, netZ = load_saved_model(f'{PATH}/runs2/nets_{rn}/netZ_nag', netZ)
		netZ = netZ.cuda()
		netG = netG.cuda()
		print(f"=> Embedding size = {len(netZ.emb.weight)}")

		if epoch > 0:
			print(f"=> Loaded successfully! epoch:{epoch}")
		else:
			raise Exception("=> No checkpoint to resume")

		Zs_real = netZ.emb.weight.data.detach().cpu().numpy()

		train_data_loader = get_loader_with_idx(train_labeled_dataset, batch_size=100,
		                                        augment=augment, shuffle=True,
		                                        offset_idx=0, offset_label=0,
		                                        autoaugment=autoaugment, **aug_param, cutout=cutout,
		                                        random_erase=random_erase)

		netZ.eval()
	save_reals=True
	if save_reals:
		# save_inter_imgs_in_npz(netZ,netG,"fid")
		# save_dataset_in_npz(test_data,"fid",name=name)
		save_dataset_in_npz(test_data,"fid")
	else:
		# for i, (idx, inputs, _) in enumerate(train_data_loader):
		# 	inputs = inputs.cuda()
		# 	targets = validate_loader_consistency(netZ, idx)
		# normalize = transforms.Normalize(mean=aug_param['mean'], std=aug_param['std'])
		# transform_train_np = transforms.Compose([
		# 		# RandomPadandCrop(32),
		# 		# RandomFlip(),
		# 		ToTensor(),
		# 		normalize,
		# 		])
		# Zs_real = netZ.emb.weight.data
		# if is_inter:
			# # z = Zs_real_numpy[idx]
			# z = Zs_real[idx]
			# z_knn_idx = np.array([random.sample(netZ.label2idx[x], 1) for x in targets])
			# ratios = list(np.linspace(0.1, 0.4, 4))
			# rnd_ratio = random.sample(ratios, 1)[0]
			# z = z.float().cuda()
			# z_knn = Zs_real[z_knn_idx[:, 0]]  # * 0.9 + Zs_real[z_knn_idx[:, 1]] * 0.1
			# inter_z = slerp_torch(rnd_ratio, z.unsqueeze(0), z_knn.cuda().float().unsqueeze(0))
			# # inter_z_slerp = interpolate_points(z, z_knn.float(), 10, print_mode=False, slerp=True)
			# # inter_z = interpolate_points(z,z_knn.float(),10,print_mode=False,slerp=False)
			# # targets = torch.tensor(targets).long().cuda()
			# # targets = targets.repeat(10)
			#
			# code = get_code(idx)
			# generated_img = netG(inter_z.squeeze().cuda(), code)
			# try:
			# 	imgs = torch.stack([normalize((img)) for img in generated_img.detach().cpu()])
			# except:
			# 	imgs = torch.stack([normalize((img)) for img in generated_img])

			# save_inter_imgs(netZ, netG, "fid")
		save_inter_imgs_in_npz(netZ,netG,keyword.split("/")[-1])


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)


if __name__ == '__main__':
	run_()
	print("Done!")
