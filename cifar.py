from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity

if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle


class CIFAR10(Dataset):
	"""`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
	Args:
		root (string): Root directory of dataset where directory
			``cifar-10-batches-py`` exists or will be saved to if download is set to True.
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transform that takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
	"""
	base_folder = 'cifar-10-batches-py'
	url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	filename = "cifar-10-python.tar.gz"
	tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
	train_list = [
			['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
			['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
			['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
			['data_batch_4', '634d18415352ddfa80567beed471001a'],
			['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
			]

	test_list = [
			['test_batch', '40351d587109b95175f43aff81a1287e'],
			]
	meta = {
			'filename': 'batches.meta',
			'key': 'label_names',
			'md5': '5ff9c542aee3614f3951f8cda6e48888',
			}

	def __init__(self, root, train=True, transform=None, target_transform=None,
	             download=False):

		super(CIFAR10, self).__init__(root, transform=transform,
		                              target_transform=target_transform)

		self.train = train  # training set or test set

		if download:
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
			                   ' You can use download=True to download it')

		if self.train:
			downloaded_list = self.train_list
		else:
			downloaded_list = self.test_list

		self.data = []
		self.targets = []

		# now load the picked numpy arrays
		for file_name, checksum in downloaded_list:
			file_path = os.path.join(self.root, self.base_folder, file_name)
			with open(file_path, 'rb') as f:
				if sys.version_info[0] == 2:
					entry = pickle.load(f)
				else:
					entry = pickle.load(f, encoding='latin1')
				self.data.append(entry['data'])
				if 'labels' in entry:
					self.targets.extend(entry['labels'])
				else:
					self.targets.extend(entry['fine_labels'])

		self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
		self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

		self._load_meta()

	def _load_meta(self):
		path = os.path.join(self.root, self.base_folder, self.meta['filename'])
		if not check_integrity(path, self.meta['md5']):
			raise RuntimeError('Dataset metadata file not found or corrupted.' +
			                   ' You can use download=True to download it')
		with open(path, 'rb') as infile:
			if sys.version_info[0] == 2:
				data = pickle.load(infile)
			else:
				data = pickle.load(infile, encoding='latin1')
			self.classes = data[self.meta['key']]
		self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)

	def _check_integrity(self):
		root = self.root
		for fentry in (self.train_list + self.test_list):
			filename, md5 = fentry[0], fentry[1]
			fpath = os.path.join(root, self.base_folder, filename)
			if not check_integrity(fpath, md5):
				return False
		return True

	def download(self):
		if self._check_integrity():
			print('Files already downloaded and verified')
			return
		download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

	def extra_repr(self):
		return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
	"""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
	This is a subclass of the `CIFAR10` Dataset.
	"""
	base_folder = 'cifar-100-python'
	url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
	filename = "cifar-100-python.tar.gz"
	tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
	train_list = [
			['train', '16019d7e3df5f24257cddd939b257f8d'],
			]

	test_list = [
			['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
			]
	meta = {
			'filename': 'meta',
			'key': 'fine_label_names',
			'md5': '7973b15100ade9c7d40fb424638fde48',
			}


import os
import os.path
import hashlib
import gzip
import errno
import tarfile
import zipfile

import torch
from torch.utils.model_zoo import tqdm
from torch._six import PY3


def gen_bar_updater():
	pbar = tqdm(total=None)

	def bar_update(count, block_size, total_size):
		if pbar.total is None and total_size:
			pbar.total = total_size
		progress_bytes = count * block_size
		pbar.update(progress_bytes - pbar.n)

	return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
	md5 = hashlib.md5()
	with open(fpath, 'rb') as f:
		for chunk in iter(lambda: f.read(chunk_size), b''):
			md5.update(chunk)
	return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
	return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
	if not os.path.isfile(fpath):
		return False
	if md5 is None:
		return True
	return check_md5(fpath, md5)


def makedir_exist_ok(dirpath):
	"""
	Python2 support for os.makedirs(.., exist_ok=True)
	"""
	try:
		os.makedirs(dirpath)
	except OSError as e:
		if e.errno == errno.EEXIST:
			pass
		else:
			raise


def download_url(url, root, filename=None, md5=None):
	"""Download a file from a url and place it in root.
	Args:
		url (str): URL to download file from
		root (str): Directory to place downloaded file in
		filename (str, optional): Name to save the file under. If None, use the basename of the URL
		md5 (str, optional): MD5 checksum of the download. If None, do not check
	"""
	from six.moves import urllib

	root = os.path.expanduser(root)
	if not filename:
		filename = os.path.basename(url)
	fpath = os.path.join(root, filename)

	makedir_exist_ok(root)

	# check if file is already present locally
	if check_integrity(fpath, md5):
		print('Using downloaded and verified file: ' + fpath)
	else:  # download the file
		try:
			print('Downloading ' + url + ' to ' + fpath)
			urllib.request.urlretrieve(
					url, fpath,
					reporthook=gen_bar_updater()
					)
		except (urllib.error.URLError, IOError) as e:
			if url[:5] == 'https':
				url = url.replace('https:', 'http:')
				print('Failed download. Trying https -> http instead.'
				      ' Downloading ' + url + ' to ' + fpath)
				urllib.request.urlretrieve(
						url, fpath,
						reporthook=gen_bar_updater()
						)
			else:
				raise e
		# check integrity of downloaded file
		if not check_integrity(fpath, md5):
			raise RuntimeError("File not found or corrupted.")


def list_dir(root, prefix=False):
	"""List all directories at a given root
	Args:
		root (str): Path to directory whose folders need to be listed
		prefix (bool, optional): If true, prepends the path to each result, otherwise
			only returns the name of the directories found
	"""
	root = os.path.expanduser(root)
	directories = list(
			filter(
					lambda p: os.path.isdir(os.path.join(root, p)),
					os.listdir(root)
					)
			)

	if prefix is True:
		directories = [os.path.join(root, d) for d in directories]

	return directories


def list_files(root, suffix, prefix=False):
	"""List all files ending with a suffix at a given root
	Args:
		root (str): Path to directory whose folders need to be listed
		suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
			It uses the Python "str.endswith" method and is passed directly
		prefix (bool, optional): If true, prepends the path to each result, otherwise
			only returns the name of the files found
	"""
	root = os.path.expanduser(root)
	files = list(
			filter(
					lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
					os.listdir(root)
					)
			)

	if prefix is True:
		files = [os.path.join(root, d) for d in files]

	return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
	"""Download a Google Drive file from  and place it in root.
	Args:
		file_id (str): id of file to be downloaded
		root (str): Directory to place downloaded file in
		filename (str, optional): Name to save the file under. If None, use the id of the file.
		md5 (str, optional): MD5 checksum of the download. If None, do not check
	"""
	# Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
	import requests
	url = "https://docs.google.com/uc?export=download"

	root = os.path.expanduser(root)
	if not filename:
		filename = file_id
	fpath = os.path.join(root, filename)

	makedir_exist_ok(root)

	if os.path.isfile(fpath) and check_integrity(fpath, md5):
		print('Using downloaded and verified file: ' + fpath)
	else:
		session = requests.Session()

		response = session.get(url, params={'id': file_id}, stream=True)
		token = _get_confirm_token(response)

		if token:
			params = {'id': file_id, 'confirm': token}
			response = session.get(url, params=params, stream=True)

		_save_response_content(response, fpath)


def _get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None


def _save_response_content(response, destination, chunk_size=32768):
	with open(destination, "wb") as f:
		pbar = tqdm(total=None)
		progress = 0
		for chunk in response.iter_content(chunk_size):
			if chunk:  # filter out keep-alive new chunks
				f.write(chunk)
				progress += len(chunk)
				pbar.update(progress - pbar.n)
		pbar.close()


def _is_tarxz(filename):
	return filename.endswith(".tar.xz")


def _is_tar(filename):
	return filename.endswith(".tar")


def _is_targz(filename):
	return filename.endswith(".tar.gz")


def _is_tgz(filename):
	return filename.endswith(".tgz")


def _is_gzip(filename):
	return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
	return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
	if to_path is None:
		to_path = os.path.dirname(from_path)

	if _is_tar(from_path):
		with tarfile.open(from_path, 'r') as tar:
			tar.extractall(path=to_path)
	elif _is_targz(from_path) or _is_tgz(from_path):
		with tarfile.open(from_path, 'r:gz') as tar:
			tar.extractall(path=to_path)
	elif _is_tarxz(from_path) and PY3:
		# .tar.xz archive only supported in Python 3.x
		with tarfile.open(from_path, 'r:xz') as tar:
			tar.extractall(path=to_path)
	elif _is_gzip(from_path):
		to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
		with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
			out_f.write(zip_f.read())
	elif _is_zip(from_path):
		with zipfile.ZipFile(from_path, 'r') as z:
			z.extractall(to_path)
	else:
		raise ValueError("Extraction of {} not supported".format(from_path))

	if remove_finished:
		os.remove(from_path)


def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
	download_root = os.path.expanduser(download_root)
	if extract_root is None:
		extract_root = download_root
	if not filename:
		filename = os.path.basename(url)

	download_url(url, download_root, filename, md5)

	archive = os.path.join(download_root, filename)
	print("Extracting {} to {}".format(archive, extract_root))
	extract_archive(archive, extract_root, remove_finished)


def iterable_to_str(iterable):
	return "'" + "', '".join([str(item) for item in iterable]) + "'"


def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
	if not isinstance(value, torch._six.string_classes):
		if arg is None:
			msg = "Expected type str, but got type {type}."
		else:
			msg = "Expected type str for argument {arg}, but got type {type}."
		msg = msg.format(type=type(value), arg=arg)
		raise ValueError(msg)

	if valid_values is None:
		return value

	if value not in valid_values:
		if custom_msg is not None:
			msg = custom_msg
		else:
			msg = ("Unknown value '{value}' for argument {arg}. "
			       "Valid values are {{{valid_values}}}.")
			msg = msg.format(value=value, arg=arg,
			                 valid_values=iterable_to_str(valid_values))
		raise ValueError(msg)

	return value
