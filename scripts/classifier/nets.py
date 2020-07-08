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

        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(4, 64, 1, kernel_size=3, padding=1, padding_mode='reflect')
        self.batchnorm1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.batchnorm2 = nn.BatchNorm2d(128, affine=False)
        self.conv3 = nn.Conv2d(128, 64, 1)
        self.batchnorm3 = nn.BatchNorm2d(64, affine= False)
        self.conv4 = nn.Conv2d(64,  self.n_classes, 1)
        self.batchnorm4 = nn.BatchNorm2d(self.n_classes, affine=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = F.max_pool2d(F.relu(x), 2)
        x = self.batchnorm2(self.conv2(x))
        x = F.max_pool2d(F.relu(x), 2)
        x = self.batchnorm3(self.conv3(x))
        x = F.max_pool2d(F.relu(x), 2)
        x = self.batchnorm4(self.conv4(x))
        x = F.max_pool2d(F.relu(x), 2)

        return x
