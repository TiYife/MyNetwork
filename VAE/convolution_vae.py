import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Pokemon(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Pokemon, self).__init__()
        self.root = root
        self.image_path = [os.path.join(root, x) for x in os.listdir(root)]
        random.shuffle(self.image_path)

        if transform is not None:
            self.transform = transform

        if train:
            self.images = self.image_path[: int(.8 * len(self.image_path))]
        else:
            self.images = self.image_path[int(.8 * len(self.image_path)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item])


class ConvolutionVae(nn.Module):
    def __init__(self, in_channels, in_size, latent_dim):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        hidden_dims = []

        modules = []

