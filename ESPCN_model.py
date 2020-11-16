import torch
import torch.nn as nn
import torch.nn.functional as F

class ESPCN(torch.nn.Module):
    def __init__(self, r):
        super(ESPCN, self).__init__()
        self.upscale = r
        # input channel =1 since only consider the luminance channel in YCbCr color space
        self.sequences = nn.Sequential(
            # (f1, n1) = (5, 64)
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            # (f2, n2) = (3, 64)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            # f3 = 3
            nn.Conv2d(32, 1*self.upscale**2, kernel_size=3, stride=1, padding=0),
            # Rearranges elements in a tensor of shape (-1, C*r^2, H, W) 
            # to a tensor of shape (-1, C, H*r, W*r)
            nn.PixelShuffle(self.upscale)
        )
        self._initialize_weights()
      
    def _initialize_weights(self):
        '''
        Xavier Initialization
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.sequences(x)
        return x