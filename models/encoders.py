from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb

##############################
#        Encoder 
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
            This encoder uses resnet-18 to extract features, and further encode them into a distribution
            similar to VAE encoder. 

            Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file
            
            Args in constructor: 
                latent_dim: latent dimension for z 
  
            Args in forward function: 
                img: image input (from domain B)

            Returns: 
                mu: mean of the latent code 
                logvar: sigma of the latent code 
        """

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)      
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

    def reparam_trick(self, mu, logvar):
        """
        Reparameterize a random perturbation in the latent space to make it back-propagatable.
        :param mu: The mean component of the latent space
        :param logvar: The logarithm of the variance of the latent space
        :return: Backpropagatable random sample from that distribution
        """
        repar = torch.normal(0, 1, logvar.shape)    # draw random samples from (0, 1) normal distribution
        new_log = 0.5 * logvar  # input logvar is logarithm of sigma^2, we need logarithm of sigma
        std = torch.exp(new_log)    # exponentiate to get value of sigma
        eps = std * repar   # apply the trick
        ret = mu + eps    # get the actual sample
        return ret
    
    