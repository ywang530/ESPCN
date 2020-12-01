import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ESPCN(torch.nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()
        self.upscale = upscale_factor

        # input channel =1 since only consider the luminance channel in YCbCr color space
        self.sequences = nn.Sequential(
            # (f1, n1) = (5, 64)
            # nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),

            # (f2, n2) = (3, 64)
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),

            # f3 = 3
            # nn.Conv2d(32, 1*(self.upscale**2), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1)),
            # Rearranges elements in a tensor of shape (-1, C*r^2, H, W) 
            # to a tensor of shape (-1, C, H*r, W*r)
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid()
        )
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m.in_channels == 32:
    #                 nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
    #                 nn.init.zeros_(m.bias.data)
    #             else:
    #                 nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
    #                 nn.init.zeros_(m.bias.data)


    def forward(self, x):
        x = self.sequences(x)
        # x = F.sigmoid(x)
        return x


