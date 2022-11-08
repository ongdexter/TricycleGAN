from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb

##############################
#        Generator 
##############################
class Generator_Unet(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
        
        Args in constructor: 
            latent_dim: latent dimension for z 
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
        
        Args in forward function: 
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns: 
            fake_B: generated image in domain B
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator_Unet, self).__init__()
        channels, self.h, self.w = img_shape

        self.enc1 = ConvBlock(channels + latent_dim, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu')
        self.enc2 = ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.enc3 = ConvBlock(128, 256, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.enc4 = ConvBlock(256, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.enc5 = ConvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.enc6 = ConvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.enc7 = ConvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')

        self.dec7 = DeconvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.dec6 = DeconvBlock(1024, 512, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.dec5 = DeconvBlock(1024, 512, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.dec4 = DeconvBlock(1024, 256, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.dec3 = DeconvBlock(512, 128, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.dec2 = DeconvBlock(256, 64, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.dec1 = DeconvBlock(128, 3, k=4, s=2, p=1, norm=False, non_linear='Tanh')

    def forward(self, x, z):
        # latent
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, z], dim=1)
        
        # downsampling
        enc1 = self.enc1(x_with_z)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)

        # upsampling
        dec6 = self.dec1(enc7)
        dec5 = self.dec2(torch.cat([dec6, enc6], dim=1))
        dec4 = self.dec3(torch.cat([dec5, enc5], dim=1))
        dec3 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec2 = self.dec5(torch.cat([dec3, enc3], dim=1))
        dec1 = self.dec6(torch.cat([dec2, enc2], dim=1))
        
        # output
        output = self.dec7(torch.cat([dec1, enc1], dim=1))
        
        return output

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []
        
        # Convolution Layer
        layers += [nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
            
        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        
        self.conv_block = nn.Sequential(* layers)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out
    
class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='relu'):
        super(DeconvBlock, self).__init__()
        layers = []
        
        # Transpose Convolution Layer
        layers += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
        
        # Non-Linearity Layer
        if non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif non_linear == 'tanh':
            layers += [nn.Tanh()]
            
        self.deconv_block = nn.Sequential(* layers)
            
    def forward(self, x):
        out = self.deconv_block(x)
        return out

if __name__ == '__main__':
    model = Generator_Unet(8, (3, 500, 500))
    print(model)