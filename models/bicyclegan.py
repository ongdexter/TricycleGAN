import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import encoders
import generators
import discriminators

class BicycleGAN(nn.Module):
    _enc_training_cfg = {
        'lr_start': 5e-3,
        'exp_lr_gamma': 0.96
    }
    _gen_training_cfg = {
        'lr_start': 5e-3,
        'exp_lr_gamma': 0.96
    }
    _dis_training_cfg = {
        'lr_start': 5e-3,
        'exp_lr_gamma': 0.96
    }
    
    def __init__(self, latent_dim, img_shape):
        super(BicycleGAN, self).__init__()
        
        # @TODO initialize properly
        self.encoder = encoders.Encoder()
        self.generator = generators.Generator_Unet()
        self.discriminator = discriminators.Discriminator()
        
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
    
    def configure_optimizers(self):
        opt_enc = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self._enc_training_cfg['lr_start']
        )
        opt_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self._gen_training_cfg['lr_start']
        )
        opt_dis = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self._dis_training_cfg['lr_start']
        )
        sch_enc = torch.optim.lr_scheduler.ExponentialLR(opt_enc, gamma=self._enc_training_cfg['exp_lr_gamma'])
        sch_gen = torch.optim.lr_scheduler.ExponentialLR(opt_enc, gamma=self._gen_training_cfg['exp_lr_gamma'])
        sch_dis = torch.optim.lr_scheduler.ExponentialLR(opt_enc, gamma=self._dis_training_cfg['exp_lr_gamma'])
        
        return [opt_enc, opt_gen, opt_dis], [sch_enc, sch_gen, sch_dis]
    