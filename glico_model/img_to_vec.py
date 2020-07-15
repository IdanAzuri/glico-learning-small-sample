import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage


class Img2Vec():

	def __init__(self, cuda=False, model='resnet18', layer='default', layer_output_size=512):
		""" Img2Vec
		:param cuda: If set to True, will run forward pass on GPU
		:param model: String name of requested model
		:param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
		:param layer_output_size: Int depicting the output size of the requested layer
		"""
		self.device = torch.device("cuda" if cuda else "cpu")
		self.layer_output_size = layer_output_size
		self.model_name = model

		self.model, self.extraction_layer = self._get_model_and_layer(model, layer)
		num_gpu = torch.cuda.device_count()
		if num_gpu > 1:
			print("Using ", num_gpu, " GPUs!")
			self.model = nn.DataParallel(self.model.cuda(), device_ids=list(range(num_gpu)))
		self.model = self.model

		self.model.eval()

		self.scaler = transforms.Resize((224, 224))
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                                      std=[0.229, 0.224, 0.225])
		self.to_tensor = transforms.ToTensor()
		self.to_pil = ToPILImage()

	def get_vec(self, img, tensor=False):
		""" Get vector embedding from PIL image
		:param img: PIL Image
		:param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
		:returns: Numpy ndarray
		"""
		import numpy as np
		# start = time.time()
		# if type(img).__module__ == np.__name__:
		if isinstance(img, (np.ndarray)):
			img = self.to_pil(img.squeeze().astype(np.uint8))
		if isinstance(img, (torch.Tensor)):
			# img = [transforms.ToPILImage()(x_) for x_ in img]
			img = self.to_pil(img.squeeze())
		image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)
		# print(f"normalize time:{time.time() - start}")
		if self.model_name == 'alexnet':
			my_embedding = torch.zeros(1, self.layer_output_size)
		else:
			my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

		def copy_data(m, i, o):
			my_embedding.copy_(o.data)

		h = self.extraction_layer.register_forward_hook(copy_data)
		h_x = self.model(image)
		h.remove()

		if tensor:
			return my_embedding
		else:
			if self.model_name == 'alexnet':
				return my_embedding.numpy()[0, :]
			else:
				return my_embedding.numpy()[0, :, 0, 0]

	def _get_model_and_layer(self, model_name, layer):
		""" Internal method for getting layer from model
		:param model_name: model name such as 'resnet-18'
		:param layer: layer as a string for resnet-18 or int for alexnet
		:returns: pytorch model, selected layer
		"""
		if model_name.startswith('resnet'):
			model = models.__dict__[model_name](pretrained=True)
			if layer == 'default':
				layer = model._modules.get('avgpool')
				self.layer_output_size = 512
			else:
				layer = model._modules.get(layer)

			return model, layer
		if model_name.startswith('dense'):
			model = models.__dict__[model_name](pretrained=True)
			if layer == 'default':
				layer = list(model.modules())[-2]
				self.layer_output_size = 512
			else:
				layer = model._modules.get(layer)

			return model, layer

		elif model_name == 'alexnet':
			model = models.alexnet(pretrained=True)
			if layer == 'default':
				layer = model.classifier[-2]
				self.layer_output_size = 4096
			else:
				layer = model.classifier[-layer]

			return model, layer

		else:
			raise KeyError('Model %s was not found' % model_name)
