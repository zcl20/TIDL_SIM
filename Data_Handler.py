# import os

import torch
# import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
# from PIL import Image, ImageOps
import random

import numpy as np

from skimage import io, exposure


# 计算两幅图像的相似度
def cal_psnr(ima0, ima1):
    mse = torch.mean((ima0 - ima1) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr


scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(48),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
un_normalize = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])

normalize2 = transforms.Normalize(mean=[0.69747254, 0.53480325, 0.68800158], std=[0.23605522, 0.27857294, 0.21456957])
un_normalize2 = transforms.Normalize(mean=[-2.9547, -1.9198, -3.20643], std=[4.2363, 3.58972, 4.66049])

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()


def get_data_loader(config):
    # dataloaders
    data_loader = load_dataset(config['TRAIN']['TRAIN_DATA_DIR'], 'train', config)
    valid_loader = load_dataset(config['TRAIN']['TRAIN_DATA_DIR'], 'valid', config)

    return data_loader, valid_loader


# input_data to dataset for training
# heretical from class Dataset
class LatticeSIMDataset(Dataset):

    def __init__(self, root, category, config):
        self.images = []
        for folder in root.split(','):
            folder_images = glob.glob(folder+'/*.tif')
            self.images.extend(folder_images)
            # print(folder_images)

        random.seed(config['SEED'])    # train and test datasets do not overlap
        random.shuffle(self.images)

        if category == 'train':
            self.images = self.images[:config['TRAIN']['NUM_TRAIN']]
        else:
            self.images = self.images[-config['TEST']['NUM_TEST']:]

        self.len = len(self.images)
        self.scale = config['TRAIN']['SCALE']
        self.task = config['TRAIN']['TASK']
        self.nch_in = config['MODEL']['IN_CHANNELS']
        self.nch_out = config['MODEL']['OUT_CHANNELS']
        self.norm = config['TRAIN']['NORM']
        self.out = config['TRAIN']['OUTPUT_DIR']

    def __getitem__(self, index):

        stack = io.imread(self.images[index])
        # n_images = stack.shape[0]
        if self.task == 'wide_raw':
            input_images = stack[[0, -4]]
        elif self.task == 'wide_raw_pattern':
            input_images = stack[[0, -4, -1]]
        elif self.task == 'raw_pattern':
            input_images = stack[[0, -1]]
        elif self.task == 'raws':
            if self.nch_in == 3:
                input_images = stack[[0, 3, 6]]
            elif self.nch_in == 4:
                input_images = stack[[0, 5, 10, 15]]
            elif self.nch_in == 5:
                input_images = stack[[0, 5, 10, 15, 20]]
            else:
                input_images = stack[:self.nch_in]
        elif self.task == 'wide':
            input_images = stack[-4]
            input_images = np.reshape(input_images, (1, stack.shape[1], stack.shape[2]))
        else:
            print('no appropriate task')
            input_images = stack[:self.nch_in]
        output_image = stack[-2]
        wide_image = stack[-4]

        # raw img from microscope, needs normalisation and correct frame ordering
        if self.norm == 'convert':
            print('Raw input assumed - converting')
            input_images = np.rot90(input_images, axes=(1, 2))
            # could be a better method to covert
            # input_images = input_images[[6, 7, 8, 3, 4, 5, 0, 1, 2]]
            for i in range(len(input_images)):
                input_images[i] = 100 / np.max(input_images[i]) * input_images[i]
        elif 'convert' in self.norm:
            fac = float(self.norm[7:])
            input_images = np.rot90(input_images, axes=(1, 2))
            # input_images = input_images[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
            for i in range(len(input_images)):
                input_images[i] = fac * 255 / np.max(input_images[i]) * input_images[i]
        input_images = input_images.astype('float') / np.max(input_images)  # used to be /255
        output_image = output_image.astype('float') / np.max(output_image)  # used to be /255
        wide_image = wide_image.astype('float') / np.max(wide_image)  # used to be /255

        if self.norm == 'adapt_hist':
            for i in range(len(input_images)):
                input_images[i] = exposure.equalize_adapthist(input_images[i], clip_limit=0.001)
            output_image = exposure.equalize_adapthist(output_image, clip_limit=0.001)
            wide_image = exposure.equalize_adapthist(wide_image, clip_limit=0.001)
            input_images = torch.tensor(input_images).float()
            output_image = torch.tensor(output_image).unsqueeze(0).float()
            wide_image = torch.tensor(wide_image).unsqueeze(0).float()

        else:
            input_images = torch.tensor(input_images).float()
            output_image = torch.tensor(output_image).float()
            wide_image = torch.tensor(wide_image).float()
            if self.nch_out == 1:
                output_image = output_image.unsqueeze(0)
            wide_image = wide_image.unsqueeze(0)
            # normalise 
            output_image = (output_image - torch.min(output_image)) / (
                        torch.max(output_image) - torch.min(output_image))
            wide_image = (wide_image - torch.min(wide_image)) / (
                        torch.max(wide_image) - torch.min(wide_image))
            if self.norm == 'minmax':
                for i in range(len(input_images)):
                    input_images[i] = (input_images[i] - torch.min(input_images[i])) / (
                                torch.max(input_images[i]) - torch.min(input_images[i]))

        return input_images, output_image, wide_image

    def __len__(self):
        return self.len


def load_dataset(root, category, config):
    dataset = LatticeSIMDataset(root, category, config)
    if category == 'train':
        data_loader = DataLoader(dataset, batch_size=config['TRAIN']['BATCH_SIZE'],
                                 shuffle=True, num_workers=config['TRAIN']['NUM_WORKERS'])
    else:
        data_loader = DataLoader(dataset, batch_size=config['TEST']['BATCH_SIZE'], shuffle=False, num_workers=0)
    return data_loader
