import warnings

import cv2
import numpy as np

from datasets.edge2shoes import Edge2Shoe
from models.discriminators import Discriminator
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
D_VAE = Discriminator().to(gpu_id)
D_LR = Discriminator().to(gpu_id)

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

            # -------------------------------
            #  Train Generator and Encoder
            # ------------------------------
            enc_tensors = encoder(edge_tensor)
            gen_tensor = generator(torch.cat(enc_tensors, dim=-1), encoder.reparam_trick(*enc_tensors))

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)
            # ----------------------------------
            dvae_tensor = D_VAE(rgb_tensor, gen_tensor)

            # ---------------------------------
            #  Train Discriminator (cLR-GAN)
            # ---------------------------------
            dlr_tensor = D_LR(rgb_tensor, gen_tensor)

            """ Optional TODO: 
				1. You may want to visualize results during training for debugging purpose
				2. Save your model every few iterations
			"""


if __name__ == "__main__":
    main()
