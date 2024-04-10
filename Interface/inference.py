import os

import numpy as np
from termcolor import colored
import cv2

import torch
import torchvision

from torch.nn import BCELoss, ReLU
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance 
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import RandomCrop, functional, GaussianBlur, Grayscale, ToPILImage, RandomGrayscale

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch.nn import Module, Sigmoid, Linear
from torch.nn import functional as F

import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

from config import update_config
from config import _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

from collections import OrderedDict

AUTHENTIC_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Au"
TAMPERED_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Tp"
GT_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_Groundtruth"

np.random.seed(12)

class forgery_detector():

	def __init__(self):

		self.img_height = 128
		self.img_width = 128
		self.quality = 90

		self.transform = Compose([Resize((self.img_height, self.img_width), interpolation=functional.InterpolationMode.BICUBIC), ToTensor(), 
									Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		# Checking if there is GPU available
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		print('device: ', device)
		self.device = device
		
		# Recod.ai model
		self.model = self.load_model()
		#resnet_dict = torch.load(os.path.join(os.path.dirname(__file__), "best_checkpoint_90.h5"), map_location=torch.device('cpu'))
		resnet_dict = torch.load(os.path.join(os.path.dirname(__file__), "best_checkpoint.h5"), map_location=device)
		cpu_resnet_dict = self.rename_keys(resnet_dict)
		self.model.load_state_dict(cpu_resnet_dict)
		self.model.eval()

		self.zero_gradients = torch.optim.SGD(self.model.parameters(), lr=1e-4)
		self.cmap = plt.get_cmap('jet')

		# TruFor model
		config.defrost()
		config.merge_from_file(f'trufor.yaml')
		config.freeze()
		print(config)
		self.trufor_model = confcmx(cfg=config)

		model_state_file = "trufor.pth.tar"
		print('=> loading model from {}'.format(model_state_file))
		checkpoint = torch.load(os.path.join(os.path.dirname(__file__), model_state_file), map_location=torch.device(device))
		self.trufor_model.load_state_dict(checkpoint['state_dict'])
		self.trufor_model = self.trufor_model.to(device)
		self.trufor_model.eval()
		
		# E2E model

		
	def __call__(self, img_path):

		# Recod.ai model preidiction
		prob_recodai = self.activations_visualization(img_path)
		#return prob

		# TruFor prediction
		img_RGB = np.array(Image.open(img_path).convert("RGB"))
		rgb = torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0
		rgb = torch.stack([rgb])

		det = None
		conf = None

		with torch.no_grad():
			rgb = rgb.to(self.device)
			pred, conf, det, npp = self.trufor_model(rgb)

		if conf is not None:
			conf = torch.squeeze(conf, 0)
			conf = torch.sigmoid(conf)[0]
			conf = conf.cpu().numpy()

		if npp is not None:
			npp = torch.squeeze(npp, 0)[0]
			npp = npp.cpu().numpy()

		if det is not None:
			det_sig = torch.sigmoid(det).item()

		pred = torch.squeeze(pred, 0)
		pred = F.softmax(pred, dim=0)[1]
		pred = pred.cpu().numpy()

		#out_dict = dict()
		#out_dict['map'] = pred
		#out_dict['imgsize'] = tuple(rgb.shape[2:])
		#if det is not None:
		#	out_dict['score'] = det_sig
		#if conf is not None:
		#	out_dict['conf'] = conf
	
		#Transforming arrays to images
		norm = mcolors.Normalize(vmin=0, vmax=1)
		cmap = plt.cm.RdBu_r
		mapped_pred = cmap(norm(pred))
		mapped_pred = (mapped_pred * 255).astype(np.uint8)

		cmap = plt.cm.gray
		mapped_conf = cmap(norm(conf))
		mapped_conf = (mapped_conf * 255).astype(np.uint8)

		return prob_recodai, det_sig, mapped_pred, mapped_conf

		

	def rename_keys(self, dict):

		keys = dict.keys()
		new_dict = OrderedDict()
		
		for k in keys:
			new_k = ".".join(k.split(".")[1:])
			new_dict[new_k] = dict[k]

		return new_dict
		
	def load_and_process(self, img_path):
		imgPIL = self.convert_to_ela_image(img_path)
		img = self.transform(imgPIL)
		return img

	def convert_to_ela_image(self, path):

		original_image = Image.open(path).convert('RGB')

		buffer = BytesIO()

		#resaved_file_name = os.path.join('temp_images', path.split("/")[-1][:-4] + 'resaved_image.jpg')   
		#original_image.save(resaved_file_name,'JPEG',quality=self.quality)
		original_image.save(buffer,'JPEG',quality=self.quality)

		# Rewind the buffer to the beginning
		buffer.seek(0)
		resaved_image = Image.open(buffer)
		#resaved_image = Image.open(resaved_file_name)
		#os.remove(resaved_file_name)

		ela_image = ImageChops.difference(original_image,resaved_image)
		
		extrema = ela_image.getextrema()
		max_difference = max([pix[1] for pix in extrema])
		if max_difference ==0:
			max_difference = 1
		scale = 255 / max_difference
		
		ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

		return ela_image

	def activations_visualization(self, img_path):

		XAI = True
		
		self.model.eval()
		self.zero_gradients.zero_grad()

		ela_image = self.convert_to_ela_image(img_path)
		ela_image_tensor = ToTensor()(ela_image)
		ela_image_tensor = np.uint8(ela_image_tensor*255)
		self.ela_image_opencv = np.moveaxis(ela_image_tensor, 0, -1)

		img = self.load_and_process(img_path)
		img = torch.stack([img])
		
		prediction = self.model(img, XAI)
		probs = torch.exp(prediction)/torch.exp(prediction).sum()

		print(colored("Probability of forgery: {:.2%}".format(probs[0][1]), "yellow"))

		# get the gradient of the output with respect to the parameters of the model
		prediction[0][1].backward()

		# pull the gradients out of the model
		gradients = self.model.get_activations_gradient()

		# pool the gradients across the channels
		pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

		# get the activations of the last convolutional layer
		activations = self.model.get_activations(img).detach()

		num_channels = activations.shape[1]
		# weight the channels by corresponding gradients
		for i in range(num_channels):
			activations[:, i, :, :] *= pooled_gradients[i]
			
		# average the channels of the activations
		heatmap = torch.mean(activations, dim=1).squeeze()

		# relu on top of the heatmap
		# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
		heatmap = ReLU()(heatmap)

		# normalize the heatmap
		eps = 1e-9
		heatmap /= torch.max(heatmap)+eps

		# Converting the heatmap to numpy array
		heatmap = heatmap.cpu().detach().numpy()

		# Resizing and blending the activation map to the original image
		#img = cv2.imread(img_path)
		#img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
		#heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
		#heatmap = np.uint8(255 * heatmap)
		#heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
		#self.superimposed_img = np.uint8(heatmap*0.4 + img*0.6) 

		imgPIL = Image.open(img_path).convert('RGB')
		width, height = imgPIL.size
		heatmap = torch.Tensor(np.moveaxis(self.cmap(heatmap)[:,:,:3], -1, 0))
		heatmap = Resize((height, width))(ToPILImage()(heatmap))
		final_image = Image.blend(imgPIL, heatmap, alpha=0.5)

		# Convert PIL image to NumPy array
		opencv_image = np.array(final_image)

		# Convert from RGB to BGR if needed
		#opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
		self.superimposed_img = opencv_image


		return probs[0][1]


	def get_ela_and_superposed(self):
		return self.ela_image_opencv, self.superimposed_img

	def load_model(self):
		
		model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		model = ResNet18(model)
		model.eval()

		return model

class ResNet18(Module):
    
	def __init__(self, model_base):
		super(ResNet18, self).__init__()

		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.relu = model_base.relu
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4
		self.avgpool = model_base.avgpool

		self.classification_layer = Linear(512, 2, bias=True)

		# placeholder for the gradients
		self.gradients = None
		
	# hook for the gradients of the activations
	def activations_hook(self, grad):
		self.gradients = grad

	def features_conv(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x) 
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x
		
	def forward(self, x, XAI=False):
		
		x = self.features_conv(x)

		# register the hook
		if XAI:
			h = x.register_hook(self.activations_hook)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		
		#prediction = Sigmoid()(self.classification_layer(x))
		prediction = self.classification_layer(x)
		
		return prediction
	
	# method for the gradient extraction
	def get_activations_gradient(self):
		return self.gradients
    
    # method for the activation exctraction
	def get_activations(self, x):
		return self.features_conv(x)
	

#if __name__ == '__main__':
#	main()
