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

def center_crop(img, dim=(128,128)):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

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
        if config.center_crop == 'yes':
            hr_image = center_crop(hr_image, (self.image_size, self.image_size))
            if config.generate_noisy == 'no':
                lr_image = center_crop(lr_image, (self.image_size, self.image_size))
        else: 
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
    
class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_lr_image_dir (str): Test dataset address for low resolution image dir.
        test_hr_image_dir (str): Test dataset address for high resolution image dir.
    """

    def __init__(self, test_lr_image_dir, test_hr_image_dir, image_size:int = 32) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.clean_image_file_names = [image_file_name for image_file_name in os.listdir(test_hr_image_dir)]
        self.noisy_image_file_names = [image_file_name for image_file_name in os.listdir(test_lr_image_dir)]
        self.clean_image_names = [os.path.join(test_hr_image_dir, image_file_name) for image_file_name in self.clean_image_file_names]
        self.noisy_image_names = [os.path.join(test_lr_image_dir, image_file_name) for image_file_name in self.clean_image_file_names]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # # How many times the high-resolution image is the low-resolution image
        # self.upscale_factor = upscale_factor
        # # Load training dataset or test dataset
        # self.mode = mode

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        clean_image = cv2.imread(self.clean_image_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        
        if config.generate_noisy == 'no':
            noisy_image = cv2.imread(self.noisy_image_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

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

    def __len__(self) -> int:
        return len(self.clean_image_file_names)
    

class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
