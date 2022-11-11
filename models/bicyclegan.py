import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BicycleGAN(nn.Module):
    def __init__(self):
        super(BicycleGAN, self).__init__()
        
    def generator_loss(self, input_img, generated_img):
        criterion = nn.L1Loss()
        loss = criterion(input_img, generated_img)
        
        return loss
    
    def discriminator_loss(self, dis_pred_real, dis_pred_fake, label):
        criterion = nn.MSELoss()
        loss_real = criterion(dis_pred_real, label)
        loss_fake = criterion(dis_pred_fake, label)
        loss = loss_real + loss_fake
        
        return loss
    
    def KL_loss(self, mu, logvar):
        loss = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - torch.ones(mu.shape) - logvar)
        return loss
        
    def forward():
        return
    
    def training_step():
        return
    
    def validation_step():
        return
        
    def training_epoch_end(self, outputs):
        return
        
    def validation_epoch_end(self, outputs):
        return
    
    