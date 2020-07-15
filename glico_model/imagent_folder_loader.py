import os
import os.path
import sys
from collections import defaultdict
from itertools import islice

from PIL import Image

try:
	from torchvision.datasets import VisionDataset
	from torchvision.datasets.folder import make_dataset, is_image_file, default_loader
except:
	from torch.utils.data import Dataset as VisionDataset, Dataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
	"""Checks if a file is an allowed extension.
	Args:
		filename (string): path to a file
		extensions (tuple of strings): extensions to consider (lowercase)
	Returns:
		bool: True if the filename ends with one of given extensions
	"""
	return filename.lower().endswith(extensions)


def is_image_file(filename):
	"""Checks if a file is an allowed image extension.
	Args:
		filename (string): path to a file
	Returns:
		bool: True if the filename ends with a known image extension
	"""
	return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None, class_cap=None, ignore_samples_by_path=None):
	images = []
	dir = os.path.expanduser(dir)
	if not ((extensions is None) ^ (is_valid_file is None)):
		raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
	if extensions is not None:
		def is_valid_file(x):
			return has_file_allowed_extension(x, extensions)
	for target in sorted(class_to_idx.keys()):
		d = os.path.join(dir, target)
		if not os.path.isdir(d):
			continue
		for root, _, fnames in sorted(os.walk(d)):
			fnames = sorted(fnames)
			if class_cap is not None:
				fnames = islice(fnames, class_cap + 1)
			for i, fname in enumerate(fnames):
				path = os.path.join(root, fname)
				# skip images if exists
				if ignore_samples_by_path is not None and path in ignore_samples_by_path.keys():
					continue
				if is_valid_file(path):
					item = (path, class_to_idx[target])
					images.append(item)

	return images


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


class DatasetFolderFromList(VisionDataset):
	def __init__(self, root, loader=default_loader, extensions=None, transform=None, target_transform=None,
	             is_valid_file=is_image_file, jsonfile=None,
	             filter_column=None, class_cap=None, offset_idx=0, ignore_samples_by_path=None):
		try:
			super(DatasetFolderFromList, self).__init__(root)
		except:
			self.root = root
		self.transform = transform
		self.target_transform = target_transform
		if jsonfile is None:
			classes, class_to_idx = self._find_classes(self.root)
		else:
			classes, class_to_idx = self._find_classes_from_list(self
			                                                     .root, jsonfile, filter_column=filter_column)
		samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, class_cap=class_cap,
		                       ignore_samples_by_path=ignore_samples_by_path)
		if len(samples) == 0:
			raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
			                                                                     "Supported extensions are: " + ",".join(
					extensions)))

		self.loader = loader
		self.extensions = extensions

		self.classes = classes
		self.class_to_idx = class_to_idx
		self.samples = samples
		self.targets = [s[1] for s in samples]
		self.path2idx = {kv[0]: i + offset_idx for (i, kv) in enumerate(self.samples)}

	def trained_samples_by_class_name(self):
		trained_samples_by_class = defaultdict(list)
		idx2class = {v: k for k, v in self.class_to_idx.items()}
		for path, class_id in self.samples:
			trained_samples_by_class[idx2class[class_id]].append(path)
		return trained_samples_by_class

	def _find_classes_from_list(self, dir, jsonfile, filter_column):
		"""
		Finds the class folders in a dataset.
		Args:
			dir (string): Root directory path.
		Returns:
			tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
		Ensures:
			No class is a subdirectory of another.
		"""
		import json

		with open(jsonfile, 'r') as f:
			self.jsonreader = json.load(f)
		# Faster and available in Python 3.5 and above
		classes = [d.name for d in os.scandir(dir) if d.is_dir()]
		filtered_classes = list(filter(lambda x: x in self.jsonreader[filter_column], classes))
		filtered_classes.sort()
		class_to_idx = {filtered_classes[i]: i for i in range(len(filtered_classes))}
		return filtered_classes, class_to_idx

	def _find_classes(self, dir):
		"""
		Finds the class folders in a dataset.

		Args:
			dir (string): Root directory path.

		Returns:
			tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

		Ensures:
			No class is a subdirectory of another.
		"""
		if sys.version_info >= (3, 5):
			# Faster and available in Python 3.5 and above
			classes = [d.name for d in os.scandir(dir) if d.is_dir()]
		else:
			classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		return classes, class_to_idx

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (sample, target) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)
		self.path2idx[path] = index
		return sample, target  # , path

	def __len__(self):
		return len(self.samples)
