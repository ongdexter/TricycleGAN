from matplotlib import pyplot as plt
from torch.utils import data
import torchvision.transforms as tf
from PIL import Image
import numpy as np
import torch
import glob
import pdb


class Edge2Shoe(data.Dataset):
    _default_tfs = {
        'edges': tf.Compose(
            tf.ToTensor()
        ),
        'rgbs': tf.Compose(
            tf.ToTensor()
        )
    }
    
    def __init__(self, img_dir):
        image_list = []
        for img_file in glob.glob(str(img_dir) + '*'):
            image_list.append(img_file)
        self.image_list = image_list
    
    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).resize((256, 128), resample=Image.BILINEAR)
        image = np.asarray(image).transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image).float()
        edge_tensor = image_tensor[:, :, :128]
        rgb_tensor = image_tensor[:, :, 128:]
        return edge_tensor, rgb_tensor
    
    def __len__(self):
        return len(self.image_list)


def visualize_batch(batch):
    """
    Visualize a batch of images from the dataloader
    :param batch: Tuple of two tensors, first for edges, second for RGB (bz, c, w, h)
    """
    edges, rgbs = batch
    n_img = edges.shape[0]
    
    fig, ax = plt.subplots(n_img, 2)
    
    for i in range(n_img):
        e = edges[i, ...] / 255.
        r = rgbs[i, ...] / 255.
        ax[i, 0].imshow(e.numpy().transpose(1, 2, 0))
        ax[i, 1].imshow(r.numpy().transpose(1, 2, 0))
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
    
    plt.show()


if __name__ == '__main__':
    img_dir = '../../edges2shoes/train/'
    dataset = Edge2Shoe(img_dir)
    loader = data.DataLoader(dataset, batch_size=4)
    for idx, data in enumerate(loader):
        edge_tensor, rgb_tensor = data
        visualize_batch(data)
        print(idx, edge_tensor.shape, rgb_tensor.shape)
