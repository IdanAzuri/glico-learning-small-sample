from __future__ import absolute_import, division, print_function, unicode_literals

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def weights_init(m):
	classname = m.__class__.__name__
	if isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
	elif classname.find('Conv') != -1:
		init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
	elif classname.find('Emb') != -1:
		init.normal_(m.weight, mean=0, std=0.01).float()


class _netZ(nn.Module):
	def __init__(self, nz, n, num_classes, data_loader=None):
		super(_netZ, self).__init__()
		self.n = n
		self.emb = nn.Embedding(self.n, nz)
		print(self.emb.weight.shape)
		self.nz = nz
		self.num_classes = num_classes
		self.data_loader = data_loader
		self.label2idx = defaultdict(set)
		self.idx2label = dict()

	def get_norm(self):
		wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
		self.emb.weight.data = \
			self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

	def set_maps_from_loader(self):
		for i_loader, dloader in enumerate(self.data_loader):
			for i, batch in enumerate(dloader):  # batch_size=1, no shuffle
				idx = batch[0].item()
				# 	# _ = maybe_cuda(batch[1])
				label = batch[2].item()
				if i_loader < 1:
					self.label2idx[label].add(idx)
				self.idx2label[idx] = label

	def forward(self, idx):
		z = self.emb(idx).squeeze()
		return z

	def cube_init(self):
		import itertools
		from utils import maybe_cuda
		end_time = time.time()
		vertices = itertools.product('01', repeat=self.nz)  # cartesian product
		# self.vertices = [tuple(int(s) for s in v) for v in vertices]
		self.vertices = sample_from_iter(vertices, self.num_classes)
		print(f"Init Cube: num classes= {self.num_classes}")
		for i_loader, dloader in enumerate(self.data_loader):
			for i, batch in enumerate(dloader):  # batch_size=1, no shuffle
				idx = maybe_cuda(batch[0])
				_ = maybe_cuda(batch[1])
				labels = batch[2]
				if i_loader < 1:
					vertex = self.vertices[labels]
					self.emb.weight.data[idx] = self.emb.weight.data[idx] + torch.tensor(vertex).float()
					self.label2idx[labels.item()].add(idx.item())
				self.idx2label[idx.item()] = labels.item()
		self.get_norm()
		print(f"init time: {(time.time() - end_time) / 60:8.2f}m")

	def resnet_init(self):
		end_time = time.time()
		from utils import maybe_cuda
		from glico_model.img_to_vec import Img2Vec
		img2vec = Img2Vec()
		# Matrix to hold the image vectors
		for i_loader, dloader in enumerate(self.data_loader):
			for i, batch in enumerate(dloader):  # batch_size=1, no shuffle
				idx = maybe_cuda(batch[0])
				imgs = maybe_cuda(batch[1])
				labels = batch[2]
				if i_loader < 1:
					vec = img2vec.get_vec(imgs.squeeze().detach().cpu().numpy())
					self.emb.weight.data[idx] = torch.tensor(vec)
					self.label2idx[labels.item()].add(idx.item())
			self.idx2label[idx.item()] = labels.item()
		self.get_norm()
		print(f"init time: {(time.time() - end_time) / 60:8.2f}m")


class _netG(nn.Module):
	def __init__(self, z_dim, img_size, n_channels, noise=False, noise_projection=False):
		super(_netG, self).__init__()
		self.sz = img_size
		self.noise_projection = noise_projection
		self.dim_im = 128 * (img_size // 4) * (img_size // 4)
		projection_size = 100
		input_dim = z_dim
		self.noise = noise
		if noise:
			input_dim += projection_size
		# self.label_embedding = nn.Embedding(n_classes, n_classes)

		self.lin_code = nn.Linear(100, projection_size, bias=False)
		self.lin_in = nn.Linear(input_dim, 1024, bias=False)
		self.bn_in = nn.BatchNorm1d(1024)
		self.lin_im = nn.Linear(1024, self.dim_im, bias=False)
		self.bn_im = nn.BatchNorm1d(self.dim_im)

		self.conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True)
		self.bn_conv = nn.BatchNorm2d(64)
		self.conv2 = nn.ConvTranspose2d(64, n_channels, 4, 2, 1, bias=True)
		self.sig = nn.Sigmoid()
		# self.nonlin = nn.SELU(True)
		self.nonlin = nn.LeakyReLU(0.2, inplace=True)

	def main(self, z):
		z = self.lin_in(z)
		# z = self.bn_in(z)
		z = self.nonlin(z)
		z = self.lin_im(z)
		# z = self.bn_im(z)
		z = self.nonlin(z)
		z = z.view(-1, 128, self.sz // 4, self.sz // 4)
		z = self.conv1(z)
		# z = self.bn_conv(z)
		z = self.nonlin(z)
		z = self.conv2(z)
		z = self.sig(z)
		return z

	def forward(self, z, code):
		zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
		z = z.div(zn)
		if self.noise_projection:
			code = self.lin_code(code)
			code = self.nonlin(code)
		if self.noise:
			z = torch.cat((z.view(z.size(0), -1), code), -1)
		output = self.main(z)
		return output


class _netG2(nn.Module):
	def __init__(self, z_dim, img_size, n_channels, noise_projection=False):
		super(_netG2, self).__init__()
		self.sz = img_size
		self.noise_projection = noise_projection
		self.dim_im = 512 * (img_size // 8) * (img_size // 8)
		projection_size = 100
		input_dim = projection_size + z_dim
		# self.label_embedding = nn.Embedding(n_classes, n_classes)

		self.lin_code = nn.Linear(100, projection_size, bias=False)
		self.lin_in = nn.Linear(input_dim, 1024, bias=False)
		self.bn_in = nn.BatchNorm1d(1024)
		self.lin_im = nn.Linear(1024, self.dim_im, bias=False)
		self.bn_im = nn.BatchNorm1d(self.dim_im)

		self.conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
		self.bn_conv1 = nn.BatchNorm2d(256)
		self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
		self.bn_conv2 = nn.BatchNorm2d(128)
		self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
		self.bn_conv3 = nn.BatchNorm2d(64)
		self.conv4 = nn.ConvTranspose2d(64, n_channels, 3, 1, 1, bias=False)
		self.tanh = nn.Tanh()
		# self.nonlin = nn.SELU(True)
		self.nonlin = nn.ReLU()

	def main(self, z):
		z = self.lin_in(z)
		# z = self.bn_in(z)
		z = self.nonlin(z)
		z = self.lin_im(z)
		z = self.bn_im(z)
		z = self.nonlin(z)
		z = z.view(-1, 512, self.sz // 8, self.sz // 8)
		z = self.conv1(z)
		z = self.bn_conv1(z)
		z = self.nonlin(z)
		z = self.conv2(z)
		z = self.bn_conv2(z)
		z = self.nonlin(z)
		z = self.conv3(z)
		z = self.bn_conv3(z)
		z = self.nonlin(z)
		z = self.conv4(z)
		z = self.tanh(z)
		return z

	def forward(self, z, code):
		zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
		z = z.div(zn)
		if self.noise_projection:
			code = self.lin_code(code)
			code = self.nonlin(code)
		z_with_label = torch.cat((z.view(z.size(0), -1), code), -1)
		output = self.main(z_with_label)
		return output


# DCGAN generator
class DCGAN_G(torch.nn.Module):
	def __init__(self, z_dim, img_size, n_channels, noise=False, noise_projection=False):
		super(DCGAN_G, self).__init__()
		main = torch.nn.Sequential()
		self.image_size = img_size
		self.z_dim = z_dim
		self.input_dim = self.z_dim
		self.channels = n_channels
		self.g_h_size = 128
		self.nn_conv = False  # Use nearest-neighbor resized convolutions instead of strided convolutions
		self.noise_projection = noise_projection
		self.projection_size = 100
		if self.noise_projection:
			print("projected noise")
			self.input_dim += self.projection_size
		# self.label_embedding = nn.Embedding(n_classes, n_classes)
		self.lin_code = nn.Linear(100, self.projection_size, bias=False)
		mult = self.image_size // 8  # count layers

		### Start block
		# Z_size random numbers

		main.add_module('Start-ConvTranspose2d',
		                torch.nn.ConvTranspose2d(self.input_dim, self.g_h_size * mult, kernel_size=4, stride=1,
		                                         padding=0, bias=False))
		main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(self.g_h_size * mult, momentum=0.001))
		main.add_module('Start-ReLU', torch.nn.ReLU())
		# Size = (G_h_size * mult) x 4 x 4

		### Middle block (Done until we reach ? x image_size/2 x image_size/2)
		i = 1
		while mult > 1:
			if self.nn_conv:
				main.add_module('Middle-UpSample [%d]' % i, torch.nn.Upsample(scale_factor=2))
				main.add_module('Middle-Conv2d [%d]' % i,
				                torch.nn.Conv2d(self.g_h_size * mult, self.g_h_size * (mult // 2), kernel_size=3,
				                                stride=1, padding=1))
			else:
				main.add_module('Middle-ConvTranspose2d [%d]' % i,
				                torch.nn.ConvTranspose2d(self.g_h_size * mult, self.g_h_size * (mult // 2),
				                                         kernel_size=4, stride=2, padding=1, bias=False))
				main.add_module('Middle-BatchNorm2d [%d]' % i,
				                torch.nn.BatchNorm2d(self.g_h_size * (mult // 2), momentum=0.001))
				main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
			# Size = (G_h_size * (mult/(2*i))) x 8 x 8
			mult = mult // 2
			i += 1

		### End block
		# Size = G_h_size x image_size/2 x image_size/2
		if self.nn_conv:
			main.add_module('End-UpSample', torch.nn.Upsample(scale_factor=2))
			main.add_module('End-Conv2d',
			                torch.nn.Conv2d(self.g_h_size, self.channels, kernel_size=3, stride=1, padding=1))
		else:
			main.add_module('End-ConvTranspose2d',
			                torch.nn.ConvTranspose2d(self.g_h_size, self.channels, kernel_size=4, stride=2,
			                                         padding=1, bias=False))
		main.add_module('End-Tanh', torch.nn.Tanh())
		# Size = n_colors x image_size x image_size
		self.nonlin = nn.LeakyReLU(0.2, inplace=True)
		self.main = main

	def forward(self, z, code):
		z = z.view(-1, self.z_dim, 1, 1)
		zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
		z = z.div(zn)
		if self.noise_projection:
			code = self.lin_code(code)
			code = self.nonlin(code)
			z_with_label = torch.cat((z.view(z.size(0), -1), code), -1)
			z = z_with_label
		z = z.view(-1, self.input_dim, 1, 1)
		output = self.main(z)
		return output.reshape(-1, 3, self.image_size, self.image_size)


# def forward(self, input):
# 	output = self.main(input)
# 	output = output.view(output.size(0), -1)
# 	output = self.classifier(output)
# 	return output.view(-1, self.num_classes)


class DCGAN_G_small(torch.nn.Module):
	def __init__(self, z_dim, img_size, n_channels, noise=False, noise_projection=False):
		super(DCGAN_G_small, self).__init__()
		self.z_dim = z_dim
		self.sz = img_size
		# self.dense = torch.nn.Linear(z_dim, 512 * self.sz // 8 * self.sz // 8)
		# self.lin_code = nn.Linear(100, projection_size, bias=False)
		self.dim_im = 512 * (img_size // 8) * (img_size // 8)
		self.non_lin = nn.LeakyReLU(0.2)
		self.noise_projection = noise_projection
		projection_size = 100
		# input_dim = z_dim
		input_dim = z_dim
		self.noise = noise
		if noise:
			input_dim += projection_size
		if noise_projection:
			input_dim = projection_size + z_dim
		# self.label_embedding = nn.Embedding(n_classes, n_classes)

		self.lin_code = nn.Linear(100, projection_size, bias=False)
		self.lin_in = nn.Linear(input_dim, 1024, bias=False)
		self.bn_in = nn.BatchNorm1d(1024)
		self.lin_im = nn.Linear(1024, self.dim_im, bias=False)
		self.bn_im = nn.BatchNorm1d(self.dim_im)
		model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
		model += [torch.nn.BatchNorm2d(256)]
		model += [torch.nn.ReLU(True)]
		model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
		model += [torch.nn.BatchNorm2d(128)]
		model += [torch.nn.ReLU(True)]
		model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
		model += [torch.nn.BatchNorm2d(64)]
		model += [torch.nn.ReLU(True)]
		model += [torch.nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1, bias=True),
		          torch.nn.Tanh()]
		self.model = torch.nn.Sequential(*model)

	def forward(self, z, code):
		zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
		z = z.div(zn)
		if self.noise_projection:
			code = self.lin_code(code)
			code = self.non_lin(code)
		if self.noise:
			z = torch.cat((z.view(z.size(0), -1), code), -1)
		z = z
		z = self.lin_in(z)
		# z = self.bn_in(z)
		z = self.non_lin(z)
		z = self.lin_im(z)
		# z = self.bn_im(z)
		z = self.non_lin(z)
		# z = z.view(-1, 128, self.sz // 8, self.sz // 8)
		# output = self.model(self.dense(z.view(-1, self.z_dim)).view(-1, 512, self.sz // 8, self.sz // 8))
		output = self.model(z.view(-1, 512, self.sz // 8, self.sz // 8))
		return output  # .reshape((-1,3,self.sz,self.sz))


class GeneratorDCGAN(nn.Module):
	def __init__(self, nz, ngf, nc):
		super().__init__()
		self.layers = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
		                                           nn.BatchNorm2d(ngf * 8),
		                                           nn.ReLU(inplace=True)),
		                             nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
		                                           nn.BatchNorm2d(ngf * 4),
		                                           nn.ReLU(inplace=True)),
		                             nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
		                                           nn.BatchNorm2d(ngf * 2),
		                                           nn.ReLU(inplace=True)),
		                             nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
		                                           nn.BatchNorm2d(ngf),
		                                           nn.ReLU(inplace=True)),
		                             nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
		                                           nn.Tanh())])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


class Classifier(nn.Module):
	def __init__(self, num_classes, dim=128):
		super(Classifier, self).__init__()
		self.num_classes = num_classes
		self.dim = dim

		# self.bn_1 = nn.BatchNorm1d(self.dim)
		self.fc1 = nn.Linear(self.dim, 512)
		self.fc2 = nn.Linear(512, 1024)
		# self.bn_2 = nn.BatchNorm1d(1024)
		self.fc3 = nn.Linear(1024, self.num_classes)
		self.nonlin = nn.LeakyReLU(0.2, inplace=False)
		self.drop = nn.Dropout(0.4)

	# self.softmax = nn.Softmax()
	# self.sig = nn.Sigmoid()

	def forward(self, x):
		batch_size = x.size(0)
		x = x.view(batch_size, self.dim)
		# x = self.bn_1(x)
		x = self.fc1(x)
		x = self.nonlin(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.nonlin(x)
		# x = self.bn_2(x)
		x = self.drop(x)
		x = self.fc3(x)
		return x


import itertools
import random


def sample_from_iter(iterable, k):
	to_int = (lambda t: tuple(int(s) for s in t))

	it = iter(iterable)
	assert k > 0, "sample size must be positive"
	l = []
	samples = list(itertools.islice(it, k))  # fill the reservoir
	random.shuffle(samples)  # if number of items less then *k* then
	for s in samples:
		l.append(np.array(to_int(s)))
	return l
