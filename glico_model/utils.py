from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import math

import numpy as np
import os
import re
import scipy
import torch
import torch.utils
import torchvision
from io import BytesIO
from operator import itemgetter
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from autoaugment import Cutout_, AutoAugment
from glico_model.model import weights_init
from vgg_arch import vgg19_bn
from wide_resnet import WideResNet
from wideresnet_2 import WideResNet2

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from glico_model import vgg_metric

try:
	import tensorflow as tf
except:
	pass

NAGParams = collections.namedtuple('NAGParams',
                                   'nz force_l2 is_pixel z_init is_classifier disc_net loss data_name noise_proj shot')
NAGParams.__new__.__defaults__ = (None, None, None, None)
GANParams = collections.namedtuple('GANParams', 'ndf weight_d')
GANParams.__new__.__defaults__ = (None, None, None)
OptParams = collections.namedtuple('OptParams', 'lr factor ' +
                                   'batch_size epochs ' +
                                   'decay_epochs decay_rate gamma')
OptParams.__new__.__defaults__ = (None, None, None, None, None, None, None)
ImageParams = collections.namedtuple('ImageParams', 'sz nc n mu sd')
ImageParams.__new__.__defaults__ = (None, None, None)


def distance_metric(sz, nc, force_l2=False):
	# return vgg_metric._VGGFixedDistance()
	if force_l2:
		return maybe_cuda(nn.L1Loss(), is_block=False)
	if sz == 16:
		return vgg_metric._VGGDistance(2)
	elif sz == 32:
		return vgg_metric._VGGDistance(3)
	elif sz == 64:
		return vgg_metric._VGGDistance(4)
	elif sz > 64:
		return vgg_metric._VGGMSDistance()


def _conv_layer(n_input, n_output, k, stride=1, padding=1, bias=True):
	"3x3 convolution with padding"
	seq = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=k, stride=stride, padding=padding, bias=bias),
	                    nn.BatchNorm2d(n_output), nn.LeakyReLU(True),
	                    nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False))
	# if Config.model.use_dropout:  # Add dropout module
	#     list_seq = list(seq.modules())[1:]
	#     list_seq.append(nn.Dropout(Config.model.dropout))
	#     seq = nn.Sequential(*list_seq)
	return seq


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_kernels(tensor, rn, epoch, num_cols=6):
	# Normalise
	maxVal = tensor.max()
	minVal = abs(tensor.min())
	maxVal = max(maxVal, minVal)
	tensor = tensor / maxVal
	tensor = tensor / 2
	tensor = tensor + 0.5
	num_rows = 1
	fig = plt.figure(figsize=(num_cols, num_rows))
	i = 0
	for t in tensor:
		ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
		pilTrans = transforms.ToPILImage()
		pilImg = pilTrans(t)
		ax1.imshow(pilImg, interpolation='none')
		ax1.axis('off')
		ax1.set_xticklabels([])
		ax1.set_yticklabels([])
		i += 1

	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	f'runs/ims_{rn}/conv1_{epoch}.png'


class cnn(nn.Module):
	"""
	A model for Mini-ImageNet classification.
	"""

	def __init__(self, num_classes, dim=64, im_size=84):
		super(cnn, self).__init__()
		self.arch = "vgg_small"
		self.n_filters = dim
		# The height and width of downsampled image
		ds_size = im_size // 2 ** 4

		self.layer1 = _conv_layer(3, self.n_filters, 3)
		# self.bn =  nn.BatchNorm2d(3)
		# self.pixel_norm = PixelNorm()
		# self.drop = nn.Dropout2d(drop_rate)
		self.lrelu = nn.LeakyReLU(0.1)
		self.layer2 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.layer3 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.layer4 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.out = nn.Sequential(nn.Linear(self.n_filters * ds_size ** 2, num_classes))
		self.softmax = nn.Softmax()
		# self.out = nn.Linear(self.n_filters * ds_size ** 2, Config.model.n_classes)

		# Initialize layers
		self.weights_init(self.layer1)
		self.weights_init(self.layer2)
		self.weights_init(self.layer3)
		self.weights_init(self.layer4)

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform_(m.weight, gain=np.sqrt(2))
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		# x = self.bn(x)
		# x=self.pixel_norm(x)
		x = self.layer1(x)
		x = self.lrelu(x)
		# x = self.drop(x)

		x = self.layer2(x)
		x = self.lrelu(x)
		# x = self.bn(x)
		# x = self.drop

		x = self.layer3(x)
		x = self.lrelu(x)
		# x = self.bn(x)

		x = self.layer4(x)
		x = self.lrelu(x)
		# x = self.drop(x)

		x = x.view(x.size(0), -1)
		x = self.out(x)
		return self.softmax(x)


def classifier(num_classes, dim=64):
	return cnn(num_classes=num_classes, dim=dim)


def sample_gaussian(x, m):
	x = x.data.numpy()
	mu = x.mean(0).squeeze()
	cov2 = np.cov(x, rowvar=0)
	z = np.random.multivariate_normal(mu, cov2, size=m)
	z_t = torch.from_numpy(z).float()
	radius = z_t.norm(2, 1).unsqueeze(1).expand_as(z_t)
	z_t = z_t / radius
	return Variable(maybe_cuda(z_t))


def maybe_cuda(tensor):
	return tensor.cuda() if torch.cuda.is_available() else tensor


class IndexToImageDataset(Dataset):
	"""Wrap a dataset to map indices to images
	In other words, instead of producing (X, y) it produces (idx, X). The label
	y is not relevant for our task.
	"""

	def __init__(self, base_dataset, transform=None, offset_idx=0, offset_label=0):
		'''

		:param base_dataset:
		:param transform:
		:param path2idx: verify the index is consistence to the path
		'''
		# assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
		self.base = base_dataset
		self.transform = transform
		self.offset_idx = offset_idx
		self.offset_label = offset_label

	def __len__(self):
		return len(self.base)

	def __getitem__(self, idx):
		# img, _ = self.base[idx]
		input = self.base[idx]
		label = input[1]
		# path = input[2]
		# print(f"{img[0].shape}, {type(img[0])}")
		if self.transform:
			img = self.transform(input[0])
		else:
			img = input[0]
		return (idx + self.offset_idx, img, label + self.offset_label)


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.std = 0

	# self.all = []

	def update(self, val, n=1):
		# self.all.append(val)
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	# self.std = sqrt(mean(abs(val - np.mean(self.all)) ** 2))
	# self.sderr = 1.96 * self.std / np.sqrt(self.count)

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, *meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def print(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# import GPUtil as GPU
#
#
# GPUs = GPU.getGPUs()
#
#
# def printm(gpu):
# 	import psutil
# 	import humanize
# 	process = psutil.Process(os.getpid())
# 	print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " | Proc size: " + humanize.naturalsize(process.memory_info().rss))
# 	print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal))

def make_image_grid(x, ngrid):
	x = x.clone().cpu()
	if pow(ngrid, 2) < x.size(0):
		grid = make_grid(x[:ngrid * ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
	else:
		grid = torch.FloatTensor(ngrid * ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
		grid[:x.size(0)].copy_(x)
		grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
	return grid


def save_image_single(x, path, imsize=512):
	from PIL import Image
	grid = make_image_grid(x, 1)
	ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
	im = Image.fromarray(ndarr)
	im = im.resize((imsize, imsize), Image.NEAREST)
	im.save(path)


def save_image_grid(x, path, imsize=512, ngrid=8):
	from PIL import Image
	grid = make_image_grid(x, ngrid)
	ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
	im = Image.fromarray(ndarr)
	im = im.resize((imsize, imsize), Image.NEAREST)
	im.save(path)


irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
	"""Make a grid of images.
	Args:
		tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
			or a list of images all of the same size.
		nrow (int, optional): Number of images displayed in each row of the grid.
			The Final grid size is (B / nrow, nrow). Default is 8.
		padding (int, optional): amount of padding. Default is 2.
		normalize (bool, optional): If True, shift the image to the range (0, 1),
			by subtracting the minimum and dividing by the maximum pixel value.
		range (tuple, optional): tuple (min, max) where min and max are numbers,
			then these numbers are used to normalize the image. By default, min and max
			are computed from the tensor.
		scale_each (bool, optional): If True, scale each image in the batch of
			images separately rather than the (min, max) over all images.
		pad_value (float, optional): Value for the padded pixels.
	Example:
		See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
	"""
	if not (torch.is_tensor(tensor) or
	        (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
		raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

	# if list of tensors, convert to a 4D mini-batch Tensor
	if isinstance(tensor, list):
		tensor = torch.stack(tensor, dim=0)

	if tensor.dim() == 2:  # single image H x W
		tensor = tensor.view(1, tensor.size(0), tensor.size(1))
	if tensor.dim() == 3:  # single image
		if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
			tensor = torch.cat((tensor, tensor, tensor), 0)
		return tensor
	if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
		tensor = torch.cat((tensor, tensor, tensor), 1)

	if normalize is True:
		tensor = tensor.clone()  # avoid modifying tensor in-place
		if range is not None:
			assert isinstance(range, tuple), \
				"range has to be a tuple (min, max) if specified. min and max are numbers"

		def norm_ip(img, min, max):
			img.clamp_(min=min, max=max)
			img.add_(-min).div_(max - min)

		def norm_range(t, range):
			if range is not None:
				norm_ip(t, range[0], range[1])
			else:
				norm_ip(t, t.min(), t.max())

		if scale_each is True:
			for t in tensor:  # loop over mini-batch dimension
				norm_range(t, range)
		else:
			norm_range(tensor, range)

	# make the mini-batch of images into a grid
	nmaps = tensor.size(0)
	xmaps = min(nrow, nmaps)
	ymaps = int(math.ceil(float(nmaps) / xmaps))
	height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
	grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
	k = 0
	for y in irange(ymaps):
		for x in irange(xmaps):
			if k >= nmaps:
				break
			grid.narrow(1, y * height + padding, height - padding) \
				.narrow(2, x * width + padding, width - padding) \
				.copy_(tensor[k])
			k = k + 1
	return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
	"""Save a given Tensor into an image file.
	Args:
		tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
			saves the tensor as a grid of images by calling ``make_grid``.
		**kwargs: Other arguments are documented in ``make_grid``.
	"""
	from PIL import Image
	tensor = tensor.cpu()
	grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
	                 normalize=normalize, range=range, scale_each=scale_each)
	ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
	im = Image.fromarray(ndarr)
	im.save(filename)


def find_latest(find_path):
	sorted_path = get_sorted_path(find_path)
	if len(sorted_path) == 0:
		return None
	return sorted_path[-1]


def get_sorted_path(find_path):
	dir_path = os.path.dirname(find_path)
	base_name = os.path.basename(find_path)
	paths = []
	for root, dirs, files in os.walk(dir_path):
		for f_name in files:
			if f_name.startswith(base_name) and f_name.endswith(".pkl"):
				paths.append(os.path.join(root, f_name))
	return sorted(paths, key=lambda x: int(re.findall("\d+", os.path.basename(x))[0]))


def load_saved_model(path, model):
	latest_path = find_latest(path)
	if latest_path is None:
		print(f"No model has been found! in {path}")
		return 0, model
	if torch.cuda.is_available():
		checkpoint = torch.load(latest_path)
	else:
		checkpoint = torch.load(latest_path, map_location='cpu')
	epoch = checkpoint['epoch']
	state_dict = checkpoint['model']

	try:
		model.load_state_dict(checkpoint['model'])
	except Exception as e:
		print(e)
		# import traceback
		# print(traceback.format_exc())
		new_state_dict = collections.OrderedDict()
		for k, v in state_dict.items():
			name = k[7:]  # remove 'module.' of dataparallel
			new_state_dict[name] = v

		model.load_state_dict(new_state_dict)
	try:
		model.label2idx = checkpoint['label2idx']
		model.idx2label = checkpoint['idx2label']
	except:
		pass
	print(f"Load checkpoints...! {latest_path}")
	return epoch, model


def save_checkpoint(path_to_save, epoch, model, maps=None, max_to_keep=1):
	sorted_path = get_sorted_path(path_to_save)
	for i in range(len(sorted_path) - max_to_keep):
		os.remove(sorted_path[i])
	full_path = f"{path_to_save}_{epoch}.pkl"
	if maps is not None:
		torch.save({"epoch": epoch, 'model': model.state_dict(), "label2idx": maps[0], "idx2label": maps[1]}, full_path)
	else:
		torch.save({"epoch": epoch, 'model': model.state_dict()}, full_path)
	print(f"Save checkpoints...! {full_path}")


class TensorBoard:
	def __init__(self, log_dir):
		"""Create a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir)

	def scalar_summary(self, tag, value, step):
		"""Log a scalar variable."""
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
		self.writer.add_summary(summary, step)

	def image_summary(self, tag, images, step):
		"""Log a list of images."""

		img_summaries = []
		for i, img in enumerate(images):
			# Write the image to a string
			s = BytesIO()
			scipy.misc.toimage(img).save(s, format="png")

			# Create an Image object
			img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=img.shape[0], width=img.shape[1])
			# Create a Summary value
			img_summaries.append(tf.Summary.Value(tag=f"{tag}/{i}", image=img_sum))

		# Create and write Summary
		summary = tf.Summary(value=img_summaries)
		self.writer.add_summary(summary, step)

	def histo_summary(self, tag, values, step, bins=1000):
		"""Log a histogram of the tensor of values."""

		# Create a histogram using numpy
		counts, bin_edges = np.histogram(values, bins=bins)

		# Fill the fields of the histogram proto
		hist = tf.HistogramProto()
		hist.min = float(np.min(values))
		hist.max = float(np.max(values))
		hist.num = int(np.prod(values.shape))
		hist.sum = float(np.sum(values))
		hist.sum_squares = float(np.sum(values ** 2))

		# Drop the start of the first bin
		bin_edges = bin_edges[1:]

		# Add bin edges and counts
		for edge in bin_edges:
			hist.bucket_limit.append(edge)
		for c in counts:
			hist.bucket.append(c)

		# Create and write Summary
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
		self.writer.add_summary(summary, step)
		self.writer.flush()

	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.scalar_summary(tag, value, step)


def get_labels(dataset):
	max_label = 100
	if hasattr(dataset, 'classes'):
		return dataset.classes
	if hasattr(dataset, 'train_labels'):
		return dataset.train_labels
	if hasattr(dataset, 'labels'):
		return dataset.labels
	if hasattr(dataset, 'targets'):
		return dataset.targets
	if hasattr(dataset, 'test_labels'):
		return dataset.test_labels
	else:
		print('No labels found! ')
	return max_label


def _load_lowshot_cifar(split, data_dir, num_shot):
	"""Load mini-imagenet from numpy's npz file format."""
	split_tag = {'train': f"train_{num_shot}_shot", 'test': f"test_{num_shot}_shot"}[split]
	dataset_path = os.path.join(data_dir, 'small-shot-{}_v3.npz'.format(split_tag))
	data = np.load(dataset_path)
	fields = data['features'], data['targets']
	print(len(data['targets']))
	print(set(data['targets']))
	return fields


def get_cub_param():
	crop_size = 224  # 448
	target_size = 256  # 512
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	return {'std': std, 'mean': mean, 'rand_crop': crop_size, 'image_size': target_size}


def get_cifar100_param():
	crop_size = 32
	target_size = 32
	mean = [0.507, 0.487, 0.441]
	std = [0.267, 0.256, 0.276]
	return {'std': std, 'mean': mean, 'rand_crop': crop_size, 'image_size': target_size}

def get_cifar10_param():
	crop_size = 32
	target_size = 32
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.247, 0.243, 0.261)
	return {'std': std, 'mean': mean, 'rand_crop': crop_size, 'image_size': target_size}

def get_imagenet_param():
	crop_size = 84
	target_size = 96
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	return {'std': std, 'mean': mean, 'rand_crop': crop_size, 'image_size': target_size}


def get_train_stl_param():
	crop_size = 96
	target_size = 96
	mean = [0.44671062, 0.43980984, 0.40664645]
	std = [0.26034098, 0.25657727, 0.27126738]
	return {'std': std, 'mean': mean, 'rand_crop': crop_size, 'image_size': target_size}


def get_train_n_unlabled_stl_param():
	crop_size = 96
	target_size = 96
	mean = [0.44087802, 0.42790631, 0.38678794]
	std = [0.26826769, 0.26104504, 0.26866837]
	return {'std': std, 'mean': mean, 'rand_crop': crop_size, 'image_size': target_size}


def get_test_stl_param():
	crop_size = 96
	target_size = 96
	mean = [0.44723063, 0.43964247, 0.40495725]
	std = [0.2605645, 0.25666146, 0.26997382]
	return {'std': std, 'mean': mean, 'rand_crop': crop_size, 'image_size': target_size}


class normalize_np(object):
	def __init__(self, mean, std):
		self.mean, self.std = [np.array(a, np.float32) for a in (mean, std)]

	def __call__(self, x):
		x -= self.mean * 255
		x *= 1.0 / (255 * self.std)
		return x

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def transpose(x, source='NHWC', target='NCHW'):
	return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
	return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
	"""Crop randomly the image.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, x):
		x = pad(x, 4)

		h, w = x.shape[1:]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		x = x[:, top: top + new_h, left: left + new_w]

		return x


class RandomFlip(object):
	"""Flip randomly the image.
	"""

	def __call__(self, x):
		if np.random.rand() < 0.5:
			x = x[:, :, ::-1]

		return x.copy()


class GaussianNoise(object):
	"""Add gaussian noise to the image.
	"""

	def __call__(self, x):
		c, h, w = x.shape
		x += np.random.randn(c, h, w) * 0.15
		return x


class ToTensor(object):
	"""Transform the image to tensor.
	"""

	def __call__(self, x):
		x = torch.from_numpy(x)
		return x


def get_loader_with_idx(dataset, batch_size, image_size, rand_crop, mean=None, std=None, num_workers=6,
                        augment=False, shuffle=True,
                        offset_idx=0,
                        offset_label=0, sampler=None, eval=False, autoaugment=False, drop_last=False, cutout=False,
                        random_erase=False):
	'''
	Note not to use normalize in NagTrainer
	'''
	if std is None:
		std = [0.267, 0.256, 0.276]
	if mean is None:
		mean = [0.507, 0.487, 0.441]
	if sampler is not None:
		shuffle = False
	normalize = transforms.Normalize(mean=mean, std=std)  # CIFAR100
	transform_list = []

	is_cifar = image_size == 32
	is_stl = image_size == 96

	if isinstance(augment, bool):
		if augment:
			print(f"augment:{augment}")
			if is_cifar:
				transform_list.append(transforms.ToPILImage())  # Comment this linr for full data
				transform_list.append(transforms.RandomCrop(rand_crop, padding=4))
				transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

			elif is_stl:
				transform_list.append(transforms.RandomCrop(image_size, padding=12))
				transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
			else:  # CUB, Places365,imagenet
				transform_list.append(transforms.RandomResizedCrop(rand_crop, scale=(0.875, 1.)))
				transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
			if autoaugment:
				transform_list.append(AutoAugment())
			if cutout:
				transform_list.append(Cutout_())
			transform_list.append(transforms.ToTensor())

		if augment:
			transform_list.append(normalize)
			if random_erase:
				transform_list.append(transforms.RandomErasing())
		elif eval:
			print(f"eval:{eval}")
			shuffle = False
			if not is_cifar:
				transform_list.append(transforms.Resize(image_size))
				transform_list.append(transforms.CenterCrop(rand_crop))
			transform_list.append(transforms.ToTensor())
			transform_list.append(normalize)
		else:  # not eval | augment is False:
			print(f"not eval | augment : TRUE")
			if not is_cifar:
				transform_list.append(transforms.Resize((image_size, image_size)))
			# transform_list.append(transforms.CenterCrop(rand_crop))
			transform_list.append(transforms.ToTensor())

		transform = transforms.Compose(transform_list)
	else:
		transform = None

	loader = torch.utils.data.DataLoader(
			IndexToImageDataset(dataset, transform=transform, offset_idx=offset_idx, offset_label=offset_label),
			batch_size=batch_size,
			shuffle=shuffle, num_workers=num_workers, pin_memory=False, sampler=sampler, drop_last=drop_last)

	print(f"=>Generated data loader, res={image_size}, workers={num_workers} transform={transform} sampler={sampler}")
	return loader


def one_hot(labels, n_classes):
	y = labels.unsqueeze(1).long()
	# One hot encoding buffer that you create out of the loop and just keep reusing
	y_onehot = torch.FloatTensor(len(y), n_classes).cuda()

	y_onehot.zero_()
	return y_onehot.scatter_(1, y, 1)


def validate_loader_consistency(netZ, idx):
	try:
		targets = itemgetter(*idx.numpy().tolist())(netZ.idx2label)
	except:
		print("Something is wrong with idx2labeldict - maybe netZ wan't loaded")
	return targets


# def get_cifar100_small(data_dir, shot):
# 	train_data_imgs, train_lables = _load_lowshot_cifar(data_dir=data_dir, split="train",
# 	                                                    num_shot=shot)
# 	train_data = CompatibleDataset(train_data_imgs, train_lables)
# 	test_data_fewshot_imgs, test_lables = _load_lowshot_cifar(data_dir=data_dir, split="test",
# 	                                                          num_shot=shot)
# 	transductive_train_data = CompatibleDataset(test_data_fewshot_imgs, test_lables)
#
# 	return train_data, transductive_train_data


def get_classifier(classes, d, pretrained):
	global cnn_
	feature_extracting = False

	def set_parameter_requires_grad(model, feature_extracting):
		if feature_extracting:
			for param in model.parameters():
				param.requires_grad = False

	if d == "vgg":
		cnn_ = vgg19_bn(num_classes=classes)
	elif d == "wideresnet":
		cnn_ = WideResNet(depth=28, num_classes=classes, widen_factor=10, dropRate=0.3)
	elif d == "wideresnet2":
		cnn_ = WideResNet2(num_classes=classes)
	# cnn_ = torch.hub.load('pytorch/vision:v0.4.2', 'wide_resnet101_2', pretrained=True)
	elif d == 'densenet':
		cnn_ = torchvision.models.densenet121(pretrained=pretrained)
		num_ftrs = cnn_.classifier.in_features
		cnn_.classifier = nn.Linear(num_ftrs, classes)
	elif d == "resnet":
		cnn_ = torchvision.models.resnet101(pretrained=pretrained)
		set_parameter_requires_grad(cnn_, feature_extracting)
		num_ftrs = cnn_.fc.in_features
		cnn_.fc = nn.Linear(num_ftrs, classes)
	elif d == "resnet50":
		cnn_ = torchvision.models.resnet50(pretrained=pretrained)
		set_parameter_requires_grad(cnn_, feature_extracting)
		num_ftrs = cnn_.fc.in_features
		cnn_.fc = nn.Linear(num_ftrs, classes)
	elif d == "resnet18":
		cnn_ = torchvision.models.resnet18(pretrained=pretrained)
		set_parameter_requires_grad(cnn_, feature_extracting)
		num_ftrs = cnn_.fc.in_features
		cnn_.fc = nn.Linear(num_ftrs, classes)
	elif d == "conv":
		cnn_ = cnn(num_classes=classes, im_size=32)  # Pixel space
		cnn_.apply(weights_init)
	return cnn_


class AugmentGaussian:
	def __init__(self, validation_stddev=25, train_stddev_rng_range=(0, 50)):
		self.validation_stddev = validation_stddev
		self.train_stddev_range = train_stddev_rng_range

	def add_gaussian(self, x):
		shape = x.size()
		(minval, maxval) = self.train_stddev_range
		train_stddev = np.random.uniform(minval / 255.0, maxval / 255.0, size=shape)
		noise = torch.randn_like(x).float().cuda()
		return x + noise * torch.tensor(train_stddev).float().cuda()

	def add_validation_noise(self, x):
		noise = torch.randn_like(x).float().cuda()
		return x + noise * torch.tensor(self.validation_stddev / 255.0).float().cuda()


class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes, smoothing=0.0, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim

	def forward(self, pred, target):
		pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
