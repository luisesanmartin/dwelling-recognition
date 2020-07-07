import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
	'''
	'''

	def __init__(self, n_classes=2):
		super(net, self).__init__()
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		self.conv1 = nn.Conv2d(4, 64, 5, stride=0)
		self.conv2 = nn.Conv2d(64, 256, 1, stride=0)
		self.conv3 = nn.Conv2d(256, 128, 1, stride=0)
		self.conv4 = nn.Conv2d(128,  n_classes, 1, stride=0)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		#x = self.logsoftmax(x)

		return x
