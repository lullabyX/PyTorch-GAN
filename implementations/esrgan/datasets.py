import glob
import random
import os
import numpy as np

import os
import queue
import threading

import cv2
import numpy as np
import torch
from skimage.util import random_noise
from torch.utils.data import DataLoader, Dataset

import config
import imgproc

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, clean_image_dir, noisy_image_dir, image_size):
        # Get all image file names in folder
        self.image_file_names = [image_file_name for image_file_name in os.listdir(clean_image_dir)]
        self.clean_image_names = [os.path.join(clean_image_dir, image_file_name) for image_file_name in self.image_file_names]
        self.noisy_image_names = [os.path.join(noisy_image_dir, image_file_name) for image_file_name in self.image_file_names]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size

    def __getitem__(self, batch_index):
        # Read a batch of image data
        clean_image = cv2.imread(self.clean_image_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if config.generate_noisy == 'no':
            noisy_image = cv2.imread(self.noisy_image_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # BGR convert to RGB
        if config.generate_noisy == 'no':
            lr_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)

        # Resize image
        if config.generate_noisy == 'no':
            lr_image = cv2.resize(lr_image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        hr_image = cv2.resize(hr_image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # Add some random noise
        if config.generate_art_noise == 'yes':
            if config.generate_noisy == 'no':
                gauss_img = random_noise(lr_image, mode='gaussian', mean=0, var=0.0005, clip=True)
            else:
                gauss_img = random_noise(hr_image, mode='gaussian', mean=0, var=0.0005, clip=True)

            # add S&P noise only for black and white image
            # salt_gauss_img = torch.tensor(random_noise(gauss_img, mode='s&p', salt_vs_pepper=0.5, amount=0.0005, clip=True))
            lr_image = random_noise(gauss_img, mode='speckle', mean=0, var=0.0005,  clip=True).astype(np.float32)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)
        hr_tensor = imgproc.image_to_tensor(hr_image, False, False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self):
        return len(self.image_file_names)
