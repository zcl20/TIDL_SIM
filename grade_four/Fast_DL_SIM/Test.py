# -*- coding: utf-8 -*-
"""
test the model
"""

# Initialise
import torch
import skimage
import numpy as np
# from numpy import pi, exp

import argparse
import os
from skimage import io, exposure
import glob
import yaml
# import torchvision.transforms as transforms
# import argparse
from skimage.metrics import structural_similarity
from Models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="Model_output/out1/Fast_dl_sim_div2k.yaml",
                        help="Path to test config file.")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    config['TEST']['TEST_DATA_DIR'] = 'Test_data/SIMdata_expdata'
    config['TEST']['NUM_TEST'] = 20
    config['TEST']['WEIGHTS'] = config['TRAIN']['OUTPUT_DIR'] + '/final.pth'
    config['TEST']['OUTPUT_DIR'] = config['TRAIN']['OUTPUT_DIR'] + '/test_results'
    config['TEST']['DEVICE'] = torch.device('cuda' if torch.cuda.is_available() and not config['TEST']['CPU']
                                            else 'cpu')

    os.makedirs('%s' % config['TEST']['OUTPUT_DIR'], exist_ok=True)
    files = glob.glob('%s/*.tif' % config['TEST']['TEST_DATA_DIR'])
    if config['TEST']['NUM_TEST'] <= len(files):
        files = files[:config['TEST']['NUM_TEST']]

    # reconstruction
    net = load_model(config)
    for n1, filename in enumerate(files):
        print('[%d/%d] Reconstructing %s' % (n1 + 1, len(files), filename))
        wide_image, ml_image = ml_lattice_sim_reconstruct(net, filename, config)
        target_image = (io.imread(filename)[-2])/255
        ml_image_uint = (255 * ml_image / np.max(ml_image)).astype('uint8')
        wide_image_uint = (255 * wide_image / np.max(wide_image)).astype('uint8')
        print('SSIM_wide=%0.4f' % structural_similarity(wide_image, target_image, data_range=1))
        print('SSIM_ml=%0.4f' % structural_similarity(ml_image, target_image, data_range=1))
        skimage.io.imsave('%s/test_wf_%d.jpg' % (config['TEST']['OUTPUT_DIR'], n1), wide_image_uint)
        skimage.io.imsave('%s/test_srml_%d.jpg' % (config['TEST']['OUTPUT_DIR'], n1), ml_image_uint)


def load_model(config):
    print('Loading model')
    print(config)
    net = get_model(config)
    print('loading checkpoint', config['TEST']['WEIGHTS'])
    checkpoint = torch.load(config['TEST']['WEIGHTS'], map_location=config['TEST']['DEVICE'])
    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # net.module.load_state_dict(state_dict)
    net.load_state_dict(state_dict)

    return net


def preprocess(stack, config):  # preprocess for images, output input_image for model and wide_image

    n_images = stack.shape[0]
    if stack.shape[1] > config['TRAIN']['IMAGE_SIZE'] or stack.shape[2] > config['TRAIN']['IMAGE_SIZE']:
        print('Over 512x512! Cropping')
        stack = stack[:, :512, :512]

    # 1 raw_image+1 wide_image
    # 1 raw_image+1 wide_image+pattern
    if n_images in (2, 3):
        wide_image = stack[1]
        input_images = stack[:n_images]

    # 1 raw_image+1 wide_image+otf+gt+pattern
    elif n_images == 5:
        wide_image = stack[1]
        if config['MODEL']['IN_CHANNELS'] == 2:
            input_images = stack[:2]
        elif config['MODEL']['IN_CHANNELS'] == 3:
            input_images = stack[[0, 1, -1]]
        else:
            print('no appropriate nch_in')
            input_images = stack[:config['MODEL']['IN_CHANNELS']]

    # n raw_images
    elif n_images in (9, 16, 25):
        wide_image = np.mean(stack, 0)
        input_images = stack[:n_images]

    # n raw_images+1 wide_image+otf+gt+pattern
    elif n_images in (13, 20, 29):
        wide_image = stack[-4]
        if config['TRAIN']['TASK'] == 'wide_raw':
            input_images = stack[[0, -4]]
        elif config['TRAIN']['TASK'] == 'wide_raw_pattern':
            input_images = stack[[0, -4, -1]]
        elif config['TRAIN']['TASK'] == 'raw_pattern':
            input_images = stack[[0, -1]]
        elif config['TRAIN']['TASK'] == 'raws':
            if config['MODEL']['IN_CHANNELS'] == 3:
                input_images = stack[[0, 3, 6]]
            elif config['MODEL']['IN_CHANNELS'] == 4:
                input_images = stack[[0, 4, 8, 12]]
            elif config['MODEL']['IN_CHANNELS'] == 5:
                input_images = stack[[0, 5, 10, 15, 20]]
            else:
                input_images = stack[:config['MODEL']['IN_CHANNELS']]
        elif config['TRAIN']['TASK'] == 'wide':
            input_images = stack[-4]
            input_images = np.reshape(input_images, (1, stack.shape[1], stack.shape[2]))
        else:
            print('no appropriate task')
            input_images = stack[:config['MODEL']['IN_CHANNELS']]

    else:
        print('no appropriate input')
        wide_image = stack[0]
        input_images = stack[:n_images]

    input_images = input_images.astype('float') / np.max(input_images)  # used to be /255
    wide_image = wide_image.astype('float') / np.max(wide_image)

    if config['TRAIN']['NORM'] == 'adapt_hist':
        for i in range(len(input_images)):
            input_images[i] = exposure.equalize_adapthist(input_images[i], clip_limit=0.001)
        wide_image = exposure.equalize_adapthist(wide_image, clip_limit=0.001)
        input_images = torch.from_numpy(input_images).float()
        wide_image = torch.from_numpy(wide_image).float()
    else:
        # normalise
        input_images = torch.from_numpy(input_images).float()
        wide_image = torch.from_numpy(wide_image).float()
        wide_image = (wide_image - torch.min(wide_image)) / (torch.max(wide_image) - torch.min(wide_image))

        if config['TRAIN']['NORM'] == 'minmax':
            for i in range(len(input_images)):
                input_images[i] = (input_images[i] - torch.min(input_images[i])) / (
                        torch.max(input_images[i]) - torch.min(input_images[i]))

    return input_images, wide_image


def ml_lattice_sim_reconstruct(model, filename, config):
    stack = io.imread(filename)
    input_image, wide_image = preprocess(stack, config)
    wide_image = wide_image.numpy()

    with torch.no_grad():
        ml_image = model(input_image.unsqueeze(0).to(config['TEST']['DEVICE']))
        ml_image = ml_image.cpu()
        ml_image = torch.clamp(ml_image, min=0, max=1)
    ml_image = ml_image.squeeze().numpy()  # squeeze the dimension (size 1)

    if config['TRAIN']['NORM'] == 'adapt_hist':
        ml_image = exposure.equalize_adapthist(ml_image, clip_limit=0.01)

    return wide_image, ml_image


if __name__ == '__main__':
    main()
