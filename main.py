import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision

import random

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		# Encoder
		self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1) 
		self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(2, padding=0)
		self.pool2 = nn.MaxPool2d(2, padding=1)

		# Decoder
		self.up1 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
		self.up2 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
		self.up3 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0)
		self.conv = nn.Conv2d(16, 1, 4, stride=1, padding=2)

	def encoder(self, image):
		conv1 = self.conv1(image)
		relu1 = F.relu(conv1) #28x28x16
		#print('relu1: ', relu1.shape)
		pool1 = self.pool1(relu1) #14x14x16
		#print('pool1: ', pool1.shape)
		conv2 = self.conv2(pool1) #14x14x8
		#print('conv2: ', conv2.shape)
		relu2 = F.relu(conv2)
		#print('relu2: ', relu2.shape)
		pool2 = self.pool1(relu2) #7x7x8
		#print('pool2: ', pool2.shape)
		conv3 = self.conv3(pool2) #7x7x8
		#print('conv3: ', conv3.shape)
		relu3 = F.relu(conv3)
		#print('relu3: ', relu3.shape)
		pool3 = self.pool2(relu3) #4x4x8
		#print('pool3: ', pool3.shape)
		pool3 = pool3.view([image.size(0), 8, 4, 4]).cuda()
		return pool3

	def decoder(self, encoding):
		up1 = self.up1(encoding) #7x7x8
		#print('up1: ', up1.shape)
		up_relu1 = F.relu(up1) 
		#print('up_relu1: ', up_relu1.shape)
		up2 = self.up2(up_relu1) #14x14x8
		#print('up2: ', up2.shape)
		up_relu2 = F.relu(up2)
		#print('up_relu2: ', up_relu2.shape)
		up3 = self.up3(up_relu2) #28x28x16
		#print('up3: ', up3.shape)
		up_relu3 = F.relu(up3)
		#print('up_relu3: ', up_relu3.shape)
		logits = self.conv(up_relu3)
		#print('logits: ', logits.shape)
		logits = F.sigmoid(logits)
		logits = logits.view([encoding.size(0), 1, 28, 28]).cuda()
		return logits

	def forward(self, image):
		encoding = self.encoder(image)
		logits = self.decoder(encoding)
		return encoding, logits
def main():
	epochs = 20
	batch_size = 200
	in_dim = 28
	lr = 0.001

	#Load data
	train_data = datasets.MNIST('~/data/mnist/', train=True, transform=transforms.ToTensor(), download=True)
	test_data = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor(), download=True)
	data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
							shuffle=True, num_workers=4, drop_last=True)
	autoencoder = Autoencoder().cuda()
	criterion = nn.BCELoss()
	size = len(train_data)
	optimizer_fn = optim.Adam
	optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
	model = train(data_loader, size, autoencoder, criterion, optimizer, num_epochs=epochs)

	test_image = random.choice(test_data)
	test_image = Variable(test_image[0].unsqueeze(0).cuda())
	_, out = model(test_image)

	# plt.subplot(0)
	# plt.imshow(test_image)
	# plt.subplot(1)
	# plt.image(out)
	# plt.subplot_tool()
	# plt.show()
	torchvision.utils.save_image(test_image.data, 'orig.png')
	torchvision.utils.save_image(out.data, 'reconst.png')

def train(data_loader, size, autoencoder, criterion, optimizer, num_epochs=20):
	print('Start training')
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs-1))
		tloss = 0.0
		for data in data_loader:
			inputs, _ = data
			optimizer.zero_grad()
			encoding, logits = autoencoder(Variable(inputs.cuda()))
			loss = criterion(logits, Variable(inputs.cuda()))
			loss.backward()
			optimizer.step()
			tloss += loss.data[0]
		epoch_loss = tloss/size
		print('Epoch loss: {:4f}'.format(epoch_loss))
	print('Complete training')
	return autoencoder
if __name__ == '__main__':
	main()




