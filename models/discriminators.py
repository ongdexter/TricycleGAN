from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        """ The discriminator used in both cVAE-GAN and cLR-GAN
            
            Args in constructor: 
                in_channels: number of channel in image (default: 3 for RGB)

            Args in forward function: 
                x: image input (real_B, fake_B)
 
            Returns: 
                discriminator output: could be a single value or a matrix depending on the type of GAN
        """

        # Patch-GAN discriminator
        super(Discriminator, self).__init__()

        # conv params
        kern_sz = 4
        conv_stride = 2
        pad_width = 1
        use_bias = False    # only applies to layers followed by batchnorm

        # leaky ReLU params
        lrelu_slope = 0.2
        lrelu_inplace = True

        # final conv params
        fconv_stride = 1

        # begin building a sequence of layers
        n_filt = 64
        layers = [
            nn.Conv2d(in_channels, n_filt, kernel_size=kern_sz, stride=conv_stride, padding=pad_width),
            nn.LeakyReLU(lrelu_slope, lrelu_inplace)
        ]

        # batch-normalized layers
        for i in range(1, 3):
            nf_mult = 2 ** i
            layers.extend([
                nn.Conv2d(
                    n_filt * nf_mult // 2,
                    n_filt * nf_mult,
                    kernel_size=kern_sz,
                    stride=conv_stride,
                    padding=pad_width,
                    bias=use_bias   # due to batch-norm coming up
                ),
                nn.BatchNorm2d(n_filt * nf_mult),
                nn.LeakyReLU(lrelu_slope, lrelu_inplace)
            ])

        # final layers - note that stride is different now
        layers.extend([
            nn.Conv2d(n_filt * nf_mult, n_filt * nf_mult * 2, kernel_size=kern_sz, stride=fconv_stride, padding=pad_width, bias=use_bias),
            nn.BatchNorm2d(n_filt * nf_mult),
            nn.LeakyReLU(lrelu_slope, lrelu_inplace)
        ])
        layers.append(
            nn.Conv2d(n_filt * nf_mult, 1, kernel_size=kern_sz, stride=fconv_stride, padding=pad_width)
        )

        # store as a sequential model
        self.core = nn.Sequential(*layers)

    def forward(self, x):
        # TODO: data is not going to be a simple tensor, it might be a tuple - what should forward be doing??
        return self.core(x)


if __name__ == '__main__':
    test_num = 0

    if test_num == 0:
        dummy_model = Discriminator()
        # TODO: need to test this somehow
        #dummy_input = torch.zeros((...))