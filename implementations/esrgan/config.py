import os

import torch

device = torch.device("cuda", 0)

upscale_factor = 1
only_test_y_channel = False

# Experiment name, easy to save weights and log files
save_model_weights = ''
while True: 
    print('Enter directory for to save model weights: ')
    save_model_weights = input()
    if not os.path.exists(save_model_weights):
        print('Directory Not Found')
        continue;
    break

# save image directory
save_image_dir = 'drive/MyDrive/Thesis/Train/'

while True: 
    print('Enter drive directory to save image:')
    save_image_dir = input()
    if os.path.exists(save_image_dir):
        print(save_image_dir)
        break
    print('Directory not found')


# train image dir
clean_image_dir = "./train_B"
noisy_image_dir = "./train_A"

while True: 
    print('Enter directory for clean image: ')
    clean_image_dir = input()
    if not os.path.exists(clean_image_dir):
        print('Clean Image Directory Not Found')
        continue;
    break

# validation image dir
test_lr_image_dir = "./val_A/"
test_hr_image_dir = "./val_B/"
while True: 
    if not os.path.exists(test_hr_image_dir):
        print('Validation CLEAN Image Directory Not Found')
        print('Enter directory for validation CLEAN image: ')
        test_hr_image_dir = input()
        continue;
    break

while True: 
    if not os.path.exists(test_lr_image_dir):
        print('Validation NOISY Image Directory Not Found')
        print('Enter directory for validation NOISY image: ')
        test_lr_image_dir = input()
        continue;
    break


# Check if noisy images needs to be generated
print('Do you want to create noisy images on the fly?(yes/no): ')
generate_noisy = 'no'
while True:
    generate_noisy = input()
    if generate_noisy == 'yes' or generate_noisy == 'no':
        break


if generate_noisy == 'no':
    while True:
        print('Enter directory for noisy image: ')
        noisy_image_dir = input()
        if not os.path.exists(noisy_image_dir):
            print('Noisy Image Directory Not Found')
            continue;
        break;

# generate artificial noise
print('Do you want to add artificial noise on the fly?(yes/no): ')
generate_art_noise = 'yes'
while True:
    generate_art_noise = input()
    if generate_art_noise == 'yes' or generate_art_noise == 'no':
        break
    

# location if pretrained model
pretrained_D_weight = '/content/drive/MyDrive/RabbiThesis/Train/ESRGAN/trainV1/SavedModel/discriminator_0.pth'
pretrained_G_weight = '/content/drive/MyDrive/RabbiThesis/Train/ESRGAN/trainV1/SavedModel/generator_0.pth'

# print frequency in validation function
valid_print_frequency = 10

# hyper params

image_size = 128
batch_size = 4
num_workers = 4


# Total num epochs
# epochs: int = 20
# epochs = int(input('Number of Epochs')) 