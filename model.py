import torchvision.transforms as transforms
import torch.nn as nn
import torch



"""
	input: Bx3x224x224
"""
class PhisheConvModel_v0(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), bias=True)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), bias=True)
		self.fc1 = nn.Linear(in_features=89888, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=128)
		self.logits = nn.Linear(in_features=128, out_features=9)
		self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

		self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
		self.relu = nn.ReLU()

	def forward(self, img):
		x = self.conv1(img)
		x = self.maxpool(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.maxpool(x)
		x = self.relu(x)

		x = x.view(x.shape[0], -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.logits(x)
		return x	

	def loss(self, img, true_label_oh):
		predictions = self.forward(img)
		return self.ce_loss(predictions, true_label_oh)

	def get_eval_transform(self):
		return transforms.Compose([
			transforms.Resize([224, 224]),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])

	def get_train_transform(self):
		return transforms.Compose([
			transforms.Resize([224, 224]),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])

	

	

class Fishe_4xConv_2xFC(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=14, kernel_size=(3, 3), bias=True)
		self.conv2 = nn.Conv2d(in_channels=14, out_channels=28, kernel_size=(3, 3), bias=True)
		self.conv3 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=(3, 3), bias=True)
		self.conv4 = nn.Conv2d(in_channels=56, out_channels=112, kernel_size=(3, 3), bias=True)
		
		self.fc1 = nn.Linear(in_features=112, out_features=112)
		self.fc2 = nn.Linear(in_features=112, out_features=56)
		self.logits = nn.Linear(in_features=56, out_features=9)
		self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

		self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.bnorm1 = nn.BatchNorm2d(14)
		self.bnorm2 = nn.BatchNorm2d(28)
		self.bnorm3 = nn.BatchNorm2d(56)
		self.bnorm4 = nn.BatchNorm2d(112)
		
		self.relu = nn.ReLU()


	def forward(self, img):
		x = self.conv1(img)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm1(x)

		x = self.conv2(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm2(x)
		
		x = self.conv3(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm3(x)
		
		x = self.conv4(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm4(x)
		

		#print(x.shape)
		x = self.avgpool(x)
		#print(x.shape)

		x = x.view(x.shape[0], -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.logits(x)
		return x

	def loss(self, img, true_label_oh):
		predictions = self.forward(img)
		return self.ce_loss(predictions, true_label_oh)

	def get_eval_transform(self):
		return transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])

	def get_train_transform(self):
		return transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])


class Fishe_6xConv_1xFC(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), bias=True)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), bias=True)
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=True)
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), bias=True)
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), bias=True)
		self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), bias=True)
		

		self.fc1 = nn.Linear(in_features=18432, out_features=442)
		self.logits = nn.Linear(in_features=442, out_features=442)
		self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

		self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
		#self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.bnorm1 = nn.BatchNorm2d(16)
		self.bnorm2 = nn.BatchNorm2d(32)
		self.bnorm3 = nn.BatchNorm2d(64)
		self.bnorm4 = nn.BatchNorm2d(128)
		self.bnorm5 = nn.BatchNorm2d(256)
		self.bnorm6 = nn.BatchNorm2d(512)
		

		self.relu = nn.ReLU()


	def forward(self, img):
		x = self.conv1(img)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm1(x)

		x = self.conv2(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm2(x)
		
		x = self.conv3(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm3(x)
		
		x = self.conv4(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm4(x)
		
		x = self.conv5(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm5(x)
		
		x = self.conv6(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.bnorm6(x)
		
		

		#print(x.shape)
		#x = self.avgpool(x)
		#print(x.shape)

		x = x.view(x.shape[0], -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.logits(x)
		return x

	def loss(self, img, true_label_oh):
		predictions = self.forward(img)
		return self.ce_loss(predictions, true_label_oh)

	def get_eval_transform(self):
		return transforms.Compose([
			transforms.Resize([800, 600]),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])

	def get_train_transform(self):
		return transforms.Compose([
			transforms.Resize([800, 600]),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		
		self.conv_1_a = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
		self.conv_2_a = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=2)
		self.conv_3_a = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

		self.avgpool = nn.AvgPool2d(kernel_size=(3, 3), stride=2)
		self.conv_1_b = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
		
		self.relu = nn.ReLU()
		self.bnorm1 = nn.BatchNorm2d(in_channels)
		self.bnorm2 = nn.BatchNorm2d(in_channels)
		self.bnorm3 = nn.BatchNorm2d(out_channels)
		self.bnorm4 = nn.BatchNorm2d(out_channels)
		
		
	
	def __call__(self, x_in):
		x_left = self.conv_1_a(x_in)
		x_left = self.bnorm1(x_left)
		x_left = self.relu(x_left)

		x_left = self.conv_2_a(x_left)
		x_left = self.bnorm2(x_left)
		x_left = self.relu(x_left)

		x_left = self.conv_3_a(x_left)
		x_left = self.bnorm3(x_left)

		# x_right = nn.AdaptiveAvgPool2d((x_left.shape[:-2]))(x_in)
		x_right = self.avgpool(x_in)
		x_right = self.conv_1_b(x_right)
		x_right = self.bnorm4(x_right)
		

		x = x_left + x_right
		x = self.relu(x)
		
		return x




class DeepFisheNetV7(nn.Module):
	def __init__(self):
		super().__init__()
		self.stem_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, bias=True)
		self.stem_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), bias=True)
		self.stem_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=True)
		
		self.res_block1 = ResidualBlock(64, 64)
		self.res_block2 = ResidualBlock(64, 256)
		self.res_block3 = ResidualBlock(256, 256)
		self.res_block4 = ResidualBlock(256, 512)
		self.res_block5 = ResidualBlock(512, 512)
		
		
		self.fc1 = nn.Linear(in_features=18432, out_features=442)
		self.fc2 = nn.Linear(in_features=442, out_features=442)
		self.logits = nn.Linear(in_features=442, out_features=442)

		self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
		self.relu = nn.ReLU()
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


	def forward(self, img):
		x = self.stem_conv1(img)
		x = self.stem_conv2(x)
		x = self.stem_conv3(x)
		x = self.relu(x)

		x = self.res_block1(x)
		x = self.res_block2(x)
		x = self.res_block3(x)
		x = self.res_block4(x)
		x = self.res_block5(x)
		
		
		#x = self.avgpool(x)
		
		x = x.view(x.shape[0], -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.logits(x)
		return x

	def loss(self, img, true_label_oh):
		predictions = self.forward(img)
		return self.ce_loss(predictions, true_label_oh)

	def get_eval_transform(self):
		return transforms.Compose([
			transforms.Resize([512, 512]),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])

	def get_train_transform(self):
		return transforms.Compose([
			transforms.Resize([512, 512]),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
			])




