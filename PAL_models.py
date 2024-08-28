import math
import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models

class QAM_OCM(nn.Module):
    def __init__(self, input_channels):
        super(QAM_OCM, self).__init__()
        
        self.vgg11 = models.vgg11()
        self.vgg11.features[0] = nn.Conv2d(in_channels=input_channels, 
                                           out_channels=64, 
                                           kernel_size=3, 
                                           stride=1, 
                                           padding=1)
        self.vgg11.classifier[6] = nn.Linear(in_features=4096, out_features=1)
        self.ocm = nn.Tanh()
        self.lam = 2.0
        self.softmax = nn.Softmax(dim=0)
        
        
    def forward(self, image, label):
        
        label = nn.functional.interpolate(label, size=image.shape[-2:])
        x = torch.cat([image, label], dim=1)
        out = self.vgg11(x)
        out = self.lam*self.ocm(out)
        out = self.softmax(out)
        
        return out.view(-1)
