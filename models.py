import math
import torch
import torch.nn as nn
from torch.nn import init

BATCHNORM_TRACK_RUNNING_STATS = False
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


class BNorm_init(nn.BatchNorm2d):
    def reset_parameters(self):
        init.uniform_(self.weight, 0, 1)
        init.zeros_(self.bias)


class Conv2d_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


def _conv_block(in_chanels, out_chanels, kernel_size, padding):
    return nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, padding=padding, bias=False), 
                         #FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.LeakyReLU())

class _conv_block2(nn.Module):
    def __init__(self, in_chanels, out_chanels, kernel_size, padding):
        super().__init__()
        
        self.conv = nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                             kernel_size=kernel_size, padding=padding, bias=True),
                                 #FeatureNorm(num_features=out_chanels, eps=0.001),
                                 nn.LeakyReLU())

    def forward(self, x):
        out = self.conv(x) + x
        return out

class FeatureNorm(nn.Module):
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(self.shape, requires_grad=True, dtype=torch.float)) if include_bias else nn.Parameter(
            torch.zeros(self.shape, requires_grad=False, dtype=torch.float))

        self.eps = eps

    def forward(self, features):
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias


class SegDecNet(nn.Module):
    def __init__(self, device, input_width, input_height, input_channels, output_shape, tt=1.0):
        super(SegDecNet, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.tt = tt
        self.volume = nn.Sequential(_conv_block(self.input_channels, 32, 5, 2),
                                    # _conv_block(32, 32, 5, 2), # Has been accidentally left out and remained the same since then
                                    nn.MaxPool2d(2),
                                    _conv_block(32, 128, 5, 2),
                                    _conv_block2(128, 128, 5, 2),
                                    _conv_block2(128, 128, 5, 2),
                                    nn.MaxPool2d(2),
                                    nn.Dropout(0.05),
                                    _conv_block(128, 128, 5, 2),
                                    _conv_block2(128, 128, 5, 2),
                                    _conv_block(128, 128, 5, 2),
                                    _conv_block2(128, 128, 5, 2),
                                    nn.MaxPool2d(2),
                                    _conv_block(128, 1024, 15, 7))

        self.seg_mask = nn.Sequential(
                                      Conv2d_init(in_channels=1024, out_channels=1, kernel_size=5, padding=2, bias=False),
                                      nn.AvgPool2d(4, stride=2, padding=0),
                                      #FeatureNorm(num_features=1, eps=0.001, include_bias=False)
                                      )
        self.Sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(1,1)

        self.device = device

    def forward(self, input, human_mask):
        volume = self.volume(input)
        
        seg_mask = self.seg_mask(volume)
        model_mask = self.Sigmoid(seg_mask).detach()
        ######################################################
        diagree = model_mask.pow(self.tt)*(1.0-human_mask) + (1.0-model_mask).pow(self.tt)*human_mask
        prediction = (diagree).sum(dim=(2,3))
            
        return prediction, seg_mask
