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
from tqdm import tqdm

# Training Configurations
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
# img_dir = '/home/zlz/BicycleGAN/datasets/edges2shoes/train/'
train_img_dir = "../edges2shoes/train/"
val_img_dir = "../edges2shoes/val/"
img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose
img_export_fmt = "input_{:03d}_{}.png"
rand_img_export_fmt = "input_{:03d}_random_sample{:02d}.png"

log_dir = "../logs/test_logs"

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

generator = Generator(latent_dim, img_shape).to(gpu_id)
encoder = Encoder(latent_dim).to(gpu_id)

# Define generator, encoder and discriminators
generator = Generator(latent_dim, img_shape).to(gpu_id)
encoder = Encoder(latent_dim).to(gpu_id)

def generate_random_valid():
    return valid - np.abs(np.random.normal(0, label_sigma, 1))[0]

# Normalize image tensor
def norm(image):
    return (image / 255.0 - 0.5) * 2.0

# Denormalize image tensor
def denorm(tensor):
    return ((tensor + 1.0) / 2.0) * 255.0

def smart_mse_loss(preds, val):
    target = val * torch.ones_like(preds, requires_grad=False)
    return mse_loss(preds, target)

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
    export_path = os.path.join(log_dir, "images")
    # if os.path.isdir(export_path) and first_write:
    #     print(f"[FATAL]: Training session ID {training_session_id} already exists!")
    #     raise Exception
    # elif not os.path.isdir(export_path) and first_write:
    #     print(f"Generating new visualizations directory: {export_path}")
    #     os.mkdir(export_path)
    #     first_write = False
    export_file = os.path.join(export_path, fname)
    cv2.imwrite(export_file, image)
    
def write_to_disk_rand(image, format_list):
    """
    Write a single image to disk. Will create directory if not present. Raises exception if global training session ID
    is taken. Also depends on global first_write.
    :param image: numpy image, should be (<img_dims>, C)-shaped, in BGR format
    :param format_list: List of parameters to add to format string.
    """
    global first_write
    fname = rand_img_export_fmt.format(*format_list)
    # export_path = os.path.join(img_export_path, training_session_id)
    export_path = os.path.join(log_dir, "images")
    # if os.path.isdir(export_path) and first_write:
    #     print(f"[FATAL]: Training session ID {training_session_id} already exists!")
    #     raise Exception
    # elif not os.path.isdir(export_path) and first_write:
    #     print(f"Generating new visualizations directory: {export_path}")
    #     os.mkdir(export_path)
    #     first_write = False
    export_file = os.path.join(export_path, fname)
    cv2.imwrite(export_file, image)



def export_train_vis(inputs, gts, outputs, epoch_num, train_val):
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
        image_out = outputs[i, ...]
        image_out = denorm(image_out)
        image_out = image_out.permute(1, 2, 0).cpu().detach().numpy()

        image_gt = gts[i, ...]
        image_gt = denorm(image_gt)
        image_gt = image_gt.permute(1, 2, 0).cpu().detach().numpy()

        # write to disk
        img_in = inputs[i, ...]
        img_in = img_in.permute(1, 2, 0).cpu().detach().numpy()
        img_in = denorm(img_in)

        write_to_disk(img_in, [epoch_num, "src"])  # model input
        write_to_disk(image_gt, [epoch_num, "ground_truth"])  # model output
        write_to_disk(image_out, [epoch_num, "output"])  # model output
        
        
def export_rand_viz(rands, epoch_num, iter_num, train_val):
    for i in range(rands.shape[0]):
        rand = rands[i, ...]
        rand = rand.permute(1, 2, 0).cpu().detach().numpy()
        rand = denorm(rand)
        write_to_disk_rand(rand, [epoch_num, iter_num])  # model rand
        

def run_inference(val_dl):
    
    real_A = None
    real_B = None
    real_A_samples = None
    real_B_samples = None

    num_viz = 10
    num_samples = 20

    for idx, data in enumerate(tqdm(val_dl)):
        ########## Process Inputs ##########
        edge_tensor, rgb_tensor = data
        edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(
            gpu_id
        )
        real_A = edge_tensor
        real_B = rgb_tensor
        
        if real_A_samples is None:
            real_A_samples = real_A
        else:
            real_A_samples = torch.cat((real_A_samples, real_A), dim=0)
        if real_B_samples is None:
            real_B_samples = real_B
        else:
            real_B_samples = torch.cat((real_B_samples, real_B), dim=0)

        # visualization
        enc_tensors = encoder(real_B_samples)
        latent_sample = encoder.reparam_trick(*enc_tensors)
        fake_B = generator(real_A_samples, latent_sample)

        for i in range(num_samples):
            rand_sample = torch.normal(0, 1, latent_sample.shape).to(latent_sample.device)
            fat_finger_B = generator(real_A_samples, rand_sample)
            export_rand_viz(fat_finger_B, idx, i, 'val')
        
        export_train_vis(
            real_A_samples,
            real_B_samples,
            fake_B,
            idx,
            'val'
        )
        if idx >= 99:
            break
    
if __name__ == '__main__':
    CHECKPOINT_PATH = '../logs/logs_v2/checkpoints/ckpt_45.pt'
    checkpoint = torch.load(CHECKPOINT_PATH)
    generator.load_state_dict(checkpoint['generator'])
    encoder.load_state_dict(checkpoint['encoder'])

    batch_size = 1
    val_dataset = Edge2Shoe(val_img_dir)
    validation_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    run_inference(validation_loader)