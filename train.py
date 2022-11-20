import warnings

import cv2
import numpy as np
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from datasets.edge2shoes import Edge2Shoe
from models.discriminators import Discriminator, TwinDiscriminator
from models.encoders import Encoder
from models.generators import Generator_Unet as Generator

warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
from utils.vis_tools import *
from datasets import *
from models import *
import argparse, os
import itertools
import torch
import time
import pdb

# Training Configurations
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
# img_dir = '/home/zlz/BicycleGAN/datasets/edges2shoes/train/'
img_dir = (
    "/home/jackfrost/Documents/business/PENN/CIS6800/final_project/edges2shoes/train/"
)
img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose

# CAUTION: numbers below are for testing only, completely arbitrary
num_epochs = 50
batch_size = 1
lr_rate = 1e-3  # Adam optimizer learning rate
betas = (0.999, 0.99)  # Adam optimizer beta 1, beta 2
lambda_pixel = 0.5  # Loss weights for pixel loss
lambda_latent = 0.5  # Loss weights for latent regression
lambda_kl = 0.5  # Loss weights for kl divergence
latent_dim = 8  # latent dimension for the encoded images from domain B
gpu_id = 0


# Normalize image tensor
def norm(image):
    return (image / 255.0 - 0.5) * 2.0


# Denormalize image tensor
def denorm(tensor):
    return ((tensor + 1.0) / 2.0) * 255.0


# Reparameterization helper function
# (You may need this helper function here or inside models.py, depending on your encoder implementation)


# Random seeds (optional)
torch.manual_seed(1)
np.random.seed(1)

# Define DataLoader
dataset = Edge2Shoe(img_dir)
loader = data.DataLoader(dataset, batch_size=batch_size)

# Loss functions
mae_loss = torch.nn.L1Loss().to(gpu_id)

# Define generator, encoder and discriminators
generator = Generator(latent_dim, img_shape).to(gpu_id)
encoder = Encoder(latent_dim).to(gpu_id)
D_VAE = TwinDiscriminator().to(gpu_id)
D_LR = TwinDiscriminator().to(gpu_id)

# Define optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=betas)
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=betas)

# For adversarial loss (optional to use)
valid = 1
fake = 0

# training visualization
img_export_path = "./train_img/"
training_session_id = "CHANGE_THIS"
img_export_fmt = "train_vis_e-{}_i-{}_{}.png"
first_write = True

def init_log_dir(log_dir):
    versions = []
    for root, dirs, subdirs in os.walk(log_dir):
        for dir in dirs:
            if 'logs_' in dir:
                versions.append(dir.split('_')[-1])
    if len(versions) == 0:
        checkpoint_dir = log_dir + '/logs_v1'
    else:
        latest_ver = int(sorted(versions)[-1].split('v')[-1]) + 1
        checkpoint_dir = log_dir + '/logs_v' + str(latest_ver)
    os.mkdir(checkpoint_dir)
    os.mkdir(checkpoint_dir + '/checkpoints')
    
    return checkpoint_dir
    
def save_checkpoints(log_dir, epoch):
    path = log_dir + '/checkpoints/ckpt_' + str(epoch) + '.pt'
    torch.save({
        'epoch': epoch,
        'generator': generator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer_E': optimizer_E.state_dict(),
        'D_VAE': D_VAE.state_dict(),
        'optimizer_D_VAE': optimizer_D_VAE.state_dict(),
        'D_LR': D_LR.state_dict(),
        'optimizer_D_LR': optimizer_D_LR.state_dict()
    }, path)

# logging
log_dir = init_log_dir('./logs')
writer = SummaryWriter(log_dir=log_dir)

def write_to_disk(image, format_list):
    """
    Write a single image to disk. Will create directory if not present. Raises exception if global training session ID
    is taken. Also depends on global first_write.
    :param image: numpy image, should be (<img_dims>, C)-shaped, in BGR format
    :param format_list: List of parameters to add to format string.
    """
    global first_write
    fname = img_export_fmt.format(*format_list)
    export_path = os.path.join(img_export_path, training_session_id)
    if os.path.isdir(export_path) and first_write:
        print(f"[FATAL]: Training session ID {training_session_id} already exists!")
        raise Exception
    elif not os.path.isdir(export_path) and first_write:
        print(f"Generating new visualizations directory: {export_path}")
        os.mkdir(export_path)
        first_write = False
    export_file = os.path.join(export_path, fname)
    cv2.imwrite(export_file, image)


def export_train_vis(model, inputs, epoch_num):
    """
    Run inference on the model, generating some outputs and storing them to disk for inspection.
    :param model: The BicycleGAN-model - should have an inference function implemented, and should be in eval mode.
    :param inputs: An input volume to run inference on, should be shaped like a batch.
    :param epoch_num: The epoch number in which the model currently is.
    """
    outputs = (
        model.inference(inputs).detach().cpu().permute(0, 3, 1).numpy()
    )  # should be Bx<img_dims>
    for i in range(outputs.shape[0]):
        image = outputs[i, ...]

        # TODO: perform some (de)-normalization and fix channel order
        image = denorm(image)
        image = image

        # write to disk
        img_in = inputs[i, ...].detach().cpu().numpy()  # TODO: fix this as well

        write_to_disk(img_in, [epoch_num, i, "src"])  # model input
        write_to_disk(image, [epoch_num, i, "gen"])  # model output



def smart_mse_loss(preds, val):
    target = val * torch.ones_like(preds, requires_grad=False)
    return mse_loss(preds, target)

def discriminator_mse_loss(real_data, fake_data, valid_label=1, fake_label=0):
    """
    Compute a discriminator loss.
    :param real_data: Real data tensor, should be the discriminator output when input is GT.
    :param fake_data: Same but for input from generator.
    :param valid_label: Numerical value for valid target.
    :param fake_label: Numerical value for valid target.
    :return: Loss tensor.
    """
    real_loss = smart_mse_loss(real_data, valid_label)
    fake_loss = smart_mse_loss(fake_data, fake_label)
    return real_loss + fake_loss


def step_discriminators(real_A, real_B):

    """----- forward passes -----"""

    # encoder and generator
    enc_tensors = encoder(real_B)
    latent_sample = encoder.reparam_trick(*enc_tensors)
    fake_B = generator(real_A, latent_sample)

    # cVAE discriminator
    dfake_vae_1, dfake_vae_2 = D_VAE(fake_B)
    dreal_vae_1, dreal_vae_2 = D_VAE(real_B)

    # cLR discriminator
    rand_sample = torch.normal(0, 1, latent_sample.shape).to(latent_sample.device)
    fat_finger_B = generator(real_A, rand_sample)
    dfake_lr_1, dfake_lr_2 = D_LR(fat_finger_B)
    dreal_lr_1, dreal_lr_2 = D_LR(real_B)

    """----- losses -----"""

    # cVAE losses - iterate over scales
    vae_loss_scale_1 = discriminator_mse_loss(dreal_vae_1, dfake_vae_1)
    vae_loss_scale_2 = discriminator_mse_loss(dreal_vae_2, dfake_vae_2)

    # cLR losses - iterate over scales
    clr_loss_scale_1 = discriminator_mse_loss(dreal_lr_1, dfake_lr_1)
    clr_loss_scale_2 = discriminator_mse_loss(dreal_lr_2, dfake_lr_2)

    # sum them all up
    disc_loss = vae_loss_scale_1 + vae_loss_scale_2 + clr_loss_scale_1 + clr_loss_scale_2

    """----- backwards pass -----"""

    # zero grad everything
    optimizer_E.zero_grad()
    optimizer_G.zero_grad()
    optimizer_D_VAE.zero_grad()
    optimizer_D_LR.zero_grad()

    # backward
    disc_loss.backward()

    # optimizer steps
    optimizer_D_LR.step()
    optimizer_D_VAE.step()


def step_gen_enc(real_A, real_B):

    # encoder and generator
    enc_tensors = encoder(real_B)
    latent_sample = encoder.reparam_trick(*enc_tensors)
    fake_B = generator(real_A, latent_sample)

    # fool the VAE discriminator
    dfake_vae_1, dfake_vae_2 = D_VAE(fake_B)
    vae_loss_scale_1 = smart_mse_loss(dfake_vae_1, valid)
    vae_loss_scale_2 = smart_mse_loss(dfake_vae_2, valid)

    # fool the cLR discriminator
    rand_sample = torch.normal(0, 1, latent_sample.shape).to(latent_sample.device)
    fat_finger_B = generator(real_A, rand_sample)
    dfake_lr_1, dfake_lr_2 = D_LR(fat_finger_B)
    clr_loss_scale_1 = smart_mse_loss(dfake_lr_1, valid)
    clr_loss_scale_2 = smart_mse_loss(dfake_lr_2, valid)

    gen_enc_loss = vae_loss_scale_1 + vae_loss_scale_2 + clr_loss_scale_1 + clr_loss_scale_2

    # KL divergence term
    KL_div = lambda_kl * torch.sum(0.5 * (enc_tensors[0] ** 2 + torch.exp(enc_tensors[1]) - enc_tensors[1] - 1))

    # image reconstruction loss
    recon_loss = lambda_pixel * torch.mean(torch.abs(fake_B - real_B))

    # sum it all up
    total_loss = gen_enc_loss + KL_div + recon_loss

    # backwards
    # zero grad everything
    optimizer_E.zero_grad()
    optimizer_G.zero_grad()
    optimizer_D_VAE.zero_grad()
    optimizer_D_LR.zero_grad()

    # backward
    total_loss.backward(retain_graph=True)

    # optimizer steps
    optimizer_E.step()
    optimizer_G.step()


    # train G-only!
    fat_enc = encoder(fat_finger_B.detach())
    latent_recon_loss = lambda_latent * torch.mean(torch.abs(fat_enc[0] - rand_sample))

    # zero grad everything
    optimizer_E.zero_grad()
    optimizer_G.zero_grad()
    optimizer_D_VAE.zero_grad()
    optimizer_D_LR.zero_grad()

    # backwards
    latent_recon_loss.backward()
    optimizer_G.step()



def main():
    # Training
    total_steps = len(loader) * num_epochs
    step = 0
    for e in range(num_epochs):
        start = time.time()
        for idx, data in enumerate(loader):
            ########## Process Inputs ##########
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(
                gpu_id
            )
            real_A = edge_tensor
            real_B = rgb_tensor

            step_discriminators(real_A, real_B)
            step_gen_enc(real_A, real_B)

            """ Optional TODO: 
				1. You may want to visualize results during training for debugging purpose
				2. Save your model every few iterations
			"""
        # save checkpoints
        if e % 5 == 0:
            save_checkpoints(log_dir, e)
                
        # tensorboard logging
        # writer.add_scalar('loss1/train', , e)
        # writer.add_scalar('loss1/val', , e)
        # writer.add_scalar('loss2/train', , e)
        # writer.add_scalar('loss2/val', , e)
        # writer.add_scalar('images/real_A', , e)
        # writer.add_scalar('images/real_B', , e)
        # writer.add_scalar('images/', , e)


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        main()
