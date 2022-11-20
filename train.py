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
    "../edges2shoes/train/"
)
img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose

# TODO: fine-tune these somehow?
num_epochs = 50
batch_size = 8
lr_rate = 2e-4  # Adam optimizer learning rate
betas = (0.5, 0.999)  # Adam optimizer beta 1, beta 2
lambda_pixel = 10  # Loss weights for pixel loss
lambda_latent = 0.5  # Loss weights for latent regression
lambda_kl = 0.01  # Loss weights for kl divergence
latent_dim = 8  # latent dimension for the encoded images from domain B
gpu_id = 0

# For adversarial loss (optional to use)
valid = 1
fake = 0
label_sigma = 0.15
label_sigma_decay = 0.9


def generate_random_valid():
    return valid - np.abs(np.random.normal(0, label_sigma, 1))[0]


# Normalize image tensor
def norm(image):
    return (image / 255.0 - 0.5) * 2.0


# Denormalize image tensor
def denorm(tensor):
    return (((tensor + 1.0) / 2.0) * 255.0)

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
                versions.append(int(dir.split('_')[-1].split('v')[-1]))
    if len(versions) == 0:
        checkpoint_dir = log_dir + '/logs_v1'
    else:
        latest_ver = sorted(versions)[-1] + 1
        checkpoint_dir = log_dir + '/logs_v' + str(latest_ver)
    os.mkdir(checkpoint_dir)
    os.mkdir(checkpoint_dir + '/checkpoints')
    os.mkdir(checkpoint_dir + '/images')
    
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
log_dir = init_log_dir('../logs')
writer = SummaryWriter(log_dir=log_dir)

# loss logging
loss_root_dir = log_dir
training_epoch_avg_losses = []
validation_epoch_avg_losses = []

def log_losses(dir_path):
    teal = np.array(training_epoch_avg_losses)
    veal = np.array(validation_epoch_avg_losses)
    np.savez(os.path.join(dir_path, "train_epoch_avg_losses.npz"), teal)
    np.savez(os.path.join(dir_path, "val_epoch_avg_losses.npz"), veal)

def write_to_disk(image, format_list):
    """
    Write a single image to disk. Will create directory if not present. Raises exception if global training session ID
    is taken. Also depends on global first_write.
    :param image: numpy image, should be (<img_dims>, C)-shaped, in BGR format
    :param format_list: List of parameters to add to format string.
    """
    global first_write
    fname = img_export_fmt.format(*format_list)
    # export_path = os.path.join(img_export_path, training_session_id)
    export_path = os.path.join(log_dir, 'images')
    # if os.path.isdir(export_path) and first_write:
    #     print(f"[FATAL]: Training session ID {training_session_id} already exists!")
    #     raise Exception
    # elif not os.path.isdir(export_path) and first_write:
    #     print(f"Generating new visualizations directory: {export_path}")
    #     os.mkdir(export_path)
    #     first_write = False
    export_file = os.path.join(export_path, fname)
    cv2.imwrite(export_file, image)


def export_train_vis(inputs, outputs, epoch_num):
    """
    Run inference on the model, generating some outputs and storing them to disk for inspection.
    :param model: The BicycleGAN-model - should have an inference function implemented, and should be in eval mode.
    :param inputs: An input volume to run inference on, should be shaped like a batch.
    :param epoch_num: The epoch number in which the model currently is.
    """
    # outputs = (
    #     model.inference(inputs).detach().cpu().permute(0, 3, 1).numpy()
    # )  # should be Bx<img_dims>
    for i in range(outputs.shape[0]):
        image = outputs[i, ...]

        # TODO: perform some (de)-normalization and fix channel order
        image = denorm(image)
        image = image.permute(1, 2, 0).cpu().detach().numpy()

        # write to disk
        img_in = inputs[i, ...]
        img_in = img_in.permute(1, 2, 0).cpu().detach().numpy()
        img_in = denorm(img_in)

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

    # random validity target to stabilize training
    rand_valid = generate_random_valid()

    # cVAE losses - iterate over scales
    vae_loss_scale_1 = discriminator_mse_loss(
        dreal_vae_1, dfake_vae_1, valid_label=rand_valid
    )
    vae_loss_scale_2 = discriminator_mse_loss(
        dreal_vae_2, dfake_vae_2, valid_label=rand_valid
    )

    # cLR losses - iterate over scales
    clr_loss_scale_1 = discriminator_mse_loss(
        dreal_lr_1, dfake_lr_1, valid_label=rand_valid
    )
    clr_loss_scale_2 = discriminator_mse_loss(
        dreal_lr_2, dfake_lr_2, valid_label=rand_valid
    )

    # sum them all up
    disc_loss = (
        vae_loss_scale_1 + vae_loss_scale_2 + clr_loss_scale_1 + clr_loss_scale_2
    )

    # print(f"\tDISC LOSS: {disc_loss}")

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

    return disc_loss.detach().cpu()


def step_gen_enc(real_A, real_B):
    # encoder and generator
    enc_tensors = encoder(real_B)
    latent_sample = encoder.reparam_trick(*enc_tensors)
    fake_B = generator(real_A, latent_sample)

    # random validity target to stabilize training
    rand_valid = generate_random_valid()

    # fool the VAE discriminator
    dfake_vae_1, dfake_vae_2 = D_VAE(fake_B)
    vae_loss_scale_1 = smart_mse_loss(dfake_vae_1, rand_valid)
    vae_loss_scale_2 = smart_mse_loss(dfake_vae_2, rand_valid)

    # fool the cLR discriminator
    rand_sample = torch.normal(0, 1, latent_sample.shape).to(latent_sample.device)
    fat_finger_B = generator(real_A, rand_sample)
    dfake_lr_1, dfake_lr_2 = D_LR(fat_finger_B)
    clr_loss_scale_1 = smart_mse_loss(dfake_lr_1, rand_valid)
    clr_loss_scale_2 = smart_mse_loss(dfake_lr_2, rand_valid)

    gen_enc_loss = (
        vae_loss_scale_1 + vae_loss_scale_2 + clr_loss_scale_1 + clr_loss_scale_2
    )

    # KL divergence term
    KL_div = lambda_kl * torch.sum(
        0.5 * (enc_tensors[0] ** 2 + torch.exp(enc_tensors[1]) - enc_tensors[1] - 1)
    )

    # image reconstruction loss
    recon_loss = lambda_pixel * torch.mean(torch.abs(fake_B - real_B))

    # sum it all up
    total_loss = gen_enc_loss + KL_div + recon_loss

    # print(f"\tTOTAL LOSS: {total_loss}")

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

    # print(f"\tLATENT RECON LOSS: {latent_recon_loss}")

    # zero grad everything
    optimizer_E.zero_grad()
    optimizer_G.zero_grad()
    optimizer_D_VAE.zero_grad()
    optimizer_D_LR.zero_grad()

    # backwards
    latent_recon_loss.backward()
    optimizer_G.step()

    return (
        gen_enc_loss.detach().cpu(),
        KL_div.detach().cpu() / lambda_kl,
        recon_loss.detach().cpu() / lambda_pixel,
        total_loss.detach().cpu(),
        latent_recon_loss.detach().cpu(),
    )


def main():
    from tqdm import tqdm

    global label_sigma

    # Training
    total_steps = len(loader) * num_epochs
    step = 0
    try:
        for e in range(num_epochs):
            start = time.time()

            # discriminator logging
            batch_disc_losses = []

            # generator/encoder logging
            batch_gen_enc_losses = []
            batch_kl_div_losses = []
            batch_recon_losses = []
            batch_total_losses = []

            # generator only logging
            batch_latent_rec_losses = []
            
            real_A = None
            real_B = None
            
            for idx, data in enumerate(tqdm(loader)):
                ########## Process Inputs ##########
                edge_tensor, rgb_tensor = data
                edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(
                    rgb_tensor
                ).to(gpu_id)
                real_A = edge_tensor
                real_B = rgb_tensor

                disc_loss = step_discriminators(real_A, real_B)
                (
                    gen_enc_loss,
                    KL_div,
                    recon_loss,
                    total_loss,
                    latent_rec_loss,
                ) = step_gen_enc(real_A, real_B)

                batch_disc_losses.append(disc_loss)

                batch_gen_enc_losses.append(gen_enc_loss)
                batch_kl_div_losses.append(KL_div)
                batch_recon_losses.append(recon_loss)
                batch_total_losses.append(total_loss)

                batch_latent_rec_losses.append(latent_rec_loss)

                # if idx > 5:
                #     break
            
            # visualization
            enc_tensors = encoder(real_B)
            latent_sample = encoder.reparam_trick(*enc_tensors)
            fake_B = generator(real_A, latent_sample)
            
            rand_sample = torch.normal(0, 1, latent_sample.shape).to(latent_sample.device)
            fat_finger_B = generator(real_A, rand_sample)
            
            export_train_vis(torch.cat((real_A[:8], real_B[:8]), dim=-1), torch.cat((fake_B[:8], fat_finger_B[:8]), dim=-1), e)
            writer.add_image('images/real_A/train', real_A[0] / 2.0 + 0.5, e)
            writer.add_image('images/real_B/train', real_B[0] / 2.0 + 0.5, e)
            writer.add_image('images/fake_B/train', fake_B[0] / 2.0 + 0.5, e)
            writer.add_image('images/fat_finger_B/train', fat_finger_B[0] / 2.0 + 0.5, e)
                        
            training_epoch_avg_losses.append(
                [
                    np.mean(batch_disc_losses),
                    np.mean(batch_gen_enc_losses),
                    np.mean(batch_kl_div_losses),
                    np.mean(batch_recon_losses),
                    np.mean(batch_total_losses),
                    np.mean(batch_latent_rec_losses),
                ]
            )
            # decay label noise each epoch
            label_sigma *= label_sigma_decay
            
            # save checkpoints
            if e % 5 == 0:
                save_checkpoints(log_dir, e)
                    
            # tensorboard logging
            writer.add_scalar('loss_disc/train', np.mean(batch_disc_losses), e)
            writer.add_scalar('loss_gen_enc/train', np.mean(batch_gen_enc_losses), e)
            writer.add_scalar('loss_kl/train', np.mean(batch_kl_div_losses), e)
            writer.add_scalar('loss_recon/train', np.mean(batch_recon_losses), e)
            writer.add_scalar('loss_latent_rec/train', np.mean(np.mean(batch_latent_rec_losses)), e)
            writer.add_scalar('loss_total/train', np.mean(batch_total_losses), e)

    finally:
        print(f'Logging losses at: {loss_root_dir}. Access with:')
        print(f"\tnp.load('{loss_root_dir}/train_epoch_avg_losses.npz', allow_pickle=True)['arr_0']")
        log_losses(loss_root_dir)


        """ Optional TODO: 
            1. You may want to visualize results during training for debugging purpose
            2. Save your model every few iterations
        """

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        main()
