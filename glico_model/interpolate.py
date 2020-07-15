import os
import random
from math import sin

import imageio
import numpy as np
import torch
from numpy.linalg import norm
from torch.utils.data import dataloader

from glico_model.utils import save_image_grid
import torchvision.utils as vutils

from utils import get_loader_with_idx


def lerp_mat(start, end, n_steps):
	vecs = []
	ratios = np.linspace(0, 1, num=n_steps)
	for ratio in ratios:
		vecs.append(torch.lerp(start, end, ratio))
	return torch.cat(vecs, 0)


# spherical linear interpolation (slerp)
def slerp(val, low, high):
	omega = np.arccos(np.clip(np.dot(low / norm(low), high / norm(high)), -1, 1))
	so = sin(omega)
	if so == 0:
		# L'Hopital's rule/LERP
		return (1.0 - val) * low + val * high
	return sin((1.0 - val) * omega) / so * low + sin(val * omega) / so * high


def slerp_torch(val, low, high):
	low_norm = low / torch.norm(low, dim=1, keepdim=True)
	high_norm = high / torch.norm(high, dim=1, keepdim=True)
	omega = torch.acos((low_norm * high_norm).sum(1))
	so = torch.sin(omega)
	res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
	return res


# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=5, slerp=True, print_mode=True):
	# interpolate ratios between the points
	assert p1.shape[0] == p2.shape[0]
	assert isinstance(p1, (torch.Tensor)), "need to be tensor"
	ratios = np.linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	if not print_mode:
		for ratio in ratios:
			v = slerp_torch(ratio, p1.squeeze(), p2.squeeze())
			vectors.append(v)
	else:
		for j in range(p1.shape[0]):
			for ratio in ratios:
				if slerp:
					v = slerp_torch(ratio, p1[j].unsqueeze(0),
					                p2[j].unsqueeze(0))  # (1 - ratio) * p1[j] + ratio * p2[j]
				else:  # linear
					v = torch.lerp(p1[j].unsqueeze(0), p2[j].unsqueeze(0), ratio)  # (1 - ratio) * p1[j] + ratio * p2[j]
				vectors.append(v)
	cat = torch.cat(vectors)
	print(cat.shape)
	return cat


def save_inter_imgs(netZ, netG, rn, name="", vis_n=8):
	Zs_real = netZ.emb.weight.data
	if not os.path.isdir("runs"):
		os.mkdir("runs")
	if not os.path.isdir("runs/ims_%s" % rn):
		os.mkdir("runs/ims_%s" % rn)
	list_of_indices = np.arange(vis_n)
	target = np.asarray([netZ.idx2label[x] for x in list_of_indices])
	z = Zs_real[list_of_indices]
	z_knn = [random.sample(netZ.label2idx[label], 1)[0] for label in target]
	inter_z = interpolate_points(z.float(), Zs_real[z_knn].cuda().float(), n_steps=vis_n)
	code = torch.cuda.FloatTensor(inter_z.size(0), 100).normal_(0, 0.15)
	generated_img = netG(inter_z.cuda(), code)
	# Ireonc = torch.from_numpy(generated_img).float()
	# save_image_grid(generated_img.data, f'runs/ims_{rn}/inter_{name}.png', ngrid=vis_n)
	vutils.save_image(generated_img.data, f'runs/ims_{rn}/inter2_{name}.png')


def save_inter_imgs_in_npz(netZ, netG, name,vis_n=10, total_imgs=10000):
	Zs_real = netZ.emb.weight.data
	if not os.path.isdir("fid_data"):
		os.mkdir("fid_data")
	# if not os.path.isdir("fid_data/ims_%s" % rn):
	# 	os.mkdir("fid_data/ims_%s" % rn)
	epochs = total_imgs // 100
	epochs //= vis_n
	imgs_list= []
	targets = []
	filtered_list = [i for i,l in netZ.idx2label.items() if l > -1]
	for e in range(1,epochs+1):
		print(f"{e}/{epochs}")
		list_of_indices = random.sample(filtered_list, 100)
		target = np.asarray([netZ.idx2label[x] for x in list_of_indices])
		z = Zs_real[list_of_indices]
		z_knn = [random.sample(netZ.label2idx[label], 1)[0] for label in target]
		inter_z = interpolate_points(z.float(), Zs_real[z_knn].cuda().float(), n_steps=vis_n)
		code = torch.cuda.FloatTensor(inter_z.size(0), 100).normal_(0, 0.15)
		generated_img = netG(inter_z.cuda(), code)
		imgs_list.append(generated_img.detach().cpu().numpy())
		targets.append(target)
		if e == 0:
			vutils.save_image(generated_img.data, f'fid_data/st.png')
	vstack = np.concatenate(imgs_list)
	targets = np.concatenate(targets)
	print(f"data size={vstack.shape}, targets={targets.shape}")
	np.savez_compressed(f"fid_data/{name}", data=vstack, labels=targets)
	print(f"Saved in {os.getcwd()}/fid_data")

def save_dataset_in_npz(dataset, rn,name="", total_imgs=10000):
	if name == "":
		name="cifar100_test_real"
	else:
		name = name.split("/")[-1]
	if not os.path.isdir("fid_data"):
		os.mkdir("fid_data")
	if not os.path.isdir("fid_data/ims_%s" % rn):
		os.mkdir("fid_data/ims_%s" % rn)
	imgs_list= []
	targets = []
	dloader = get_loader_with_idx(dataset, batch_size=100,
	                                        augment=False, shuffle=True,
	                                        offset_idx=0, offset_label=0,image_size=32,rand_crop=32)
	for i, (idx,inputs, target) in enumerate(dloader):
		imgs_list.append(inputs)
		targets.append(target)
		# if i == 0:
		# 	print(inputs.shape)
			# vutils.save_image(np.transpose(inputs, (1, 2, 0)), f'c.png')
		if len(targets)>total_imgs:
			break
	vstack = np.concatenate(imgs_list)
	# vutils.save_image(torch.tensor(vstack[:1].squeeze()), f'fid_data/real.png')
	targets = np.concatenate(targets)
	print(f"data size={vstack.shape}, targets={targets.shape}")
	np.savez_compressed(f"fid_data/{name}", data=vstack, labels=targets)
	print(f"Saved in {os.getcwd()}/fid_data")

def tsne_vis(netZ, rn, img_size, real_imgs):
	import matplotlib.pyplot as plt
	from MulticoreTSNE import MulticoreTSNE as TSNE

	Zs_real = netZ.emb.weight.data.detach().cpu().numpy()
	if not os.path.isdir("runs"):
		os.mkdir("runs")
	if not os.path.isdir("runs/ims_%s" % rn):
		os.mkdir("runs/ims_%s" % rn)
	tsne = TSNE(n_components=2, perplexity=30, n_jobs=20)
	n_samples = len(real_imgs)
	targets = np.asarray([netZ.idx2label[x] for x in range(n_samples)])
	filtered_indices = targets[targets < 11]
	targets = targets[filtered_indices]
	Z_filter = Zs_real[filtered_indices]
	print(len(Z_filter))
	reduced_data = tsne.fit_transform(np.asarray(Z_filter, dtype='float64'))
	plot_by_latent(reduced_data, real_imgs, indices=filtered_indices, img_size=img_size, rn=rn, title="G2")
	# print(indices)
	# y_for_plot = np.concatenate([a, moves_,y_labels])
	# N = len(y_lables)
	# Y=Y[indices]
	# Y = bh_sne(np.asarray(s_t[0:N], dtype='float64'))
	# normalize
	min_1 = reduced_data[:, 0].min()
	max_1 = reduced_data[:, 0].max()
	min_2 = reduced_data[:, 1].min()
	max_2 = reduced_data[:, 1].max()
	Yn = reduced_data[:]
	Yn[:, 0] = (reduced_data[:, 0] - min_1) / (max_1 - min_1)
	Yn[:, 1] = (reduced_data[:, 1] - min_2) / (max_2 - min_2)

	## plot distribution

	unique_classes = len(np.unique(targets))

	y_labels_colors = targets

	plt.scatter(Yn[:, 1], -Yn[:, 0], c=y_labels_colors, cmap=plt.cm.get_cmap("tab20", unique_classes), s=10,
	            edgecolors='k')
	mn = int(np.floor(y_labels_colors.min()))  # colorbar min value
	mx = int(np.ceil(y_labels_colors.max()))  # colorbar max value
	md = (mx - mn) // 2
	cbar = plt.colorbar()
	cbar.set_ticks([mn, md, mx])
	cbar.set_ticklabels([mn, md, mx])
	# plt.scatter(Yn[Zs_real, 1], -Yn[Zs_real, 0], c="black", s=100, edgecolors='k', marker="x")
	# plt.scatter(Yn[indices[0], 1], -Yn[indices[0], 0], c="darkorange", s=100, edgecolors='k', marker="P", label="start")
	# plt.scatter(Yn[indices[1], 1], -Yn[indices[1], 0], c="yellow", s=100, edgecolors='k', marker="p", label="target")
	# plot_path(moves_knn, start_point_plt, targ_point_plt, title="RNN miniImagenet", more="")
	plt.savefig(f"runs/ims_{rn}/tsne_{rn}.jpg")


def plot_by_latent(reduced_data, real_imgs, indices, img_size, rn, title="G", emb_size=2000):
	import matplotlib.pyplot as plt
	# set min point to 0 and scale
	x = reduced_data - np.min(reduced_data)
	x = x / np.max(x)
	# create embedding image
	G = np.zeros((emb_size, emb_size, 3), dtype='uint8')
	# G_norm = np.zeros((S, S, 3), dtype='uint8')
	for i in indices:  # range(indices):
		# set location
		a = np.ceil(x[i, 0] * (emb_size - img_size - 1) + 1)
		b = np.ceil(x[i, 1] * (emb_size - img_size - 1) + 1)
		a = int(a - np.mod(a - 1, img_size) + 1)
		b = int(b - np.mod(b - 1, img_size) + 1)
		if G[a, b, 0] != 0:
			continue
		G[a:a + img_size, b:b + img_size, :] = real_imgs[i]  # .mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
	# imageio.imwrite(f'runs/ims_{run_name}/tst.jpg',train_data[i][0].byte().permute(1, 2, 0).numpy())
	# imageio.imwrite(f'runs/ims_{run_name}/{title}_{emb_size}.jpg', G)
	fig = plt.Figure()
	fig.patch.set_facecolor('white')

	plt.imsave(f'runs/ims_{rn}/{title}_{emb_size}.jpg', G)
	plt.close()
