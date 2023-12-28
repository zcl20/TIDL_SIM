# -*- coding: utf-8 -*-

import sys
import argparse
# import numpy as np
from numpy import pi, cos, sin, sqrt, arccos, exp
# import cv2
import os
import glob
from skimage import io, transform
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from multiprocessing import Pool
from System_Parameter import *
import random

# set image system parameter
lambda_illu = 0.515
lambda_ex = 0.488
fco = 1.25 / lambda_ex  # not full entrance na 1.25
lattice = 'square'  # hexagonal, square， line
data = 'train'

# read images
out_dir = "Test_data/SIMdata_mydata_"+lattice
os.makedirs(out_dir, exist_ok=True)
# return all .png files
# files = glob.glob("Train_data/DIV2K_subset/*.png")
files = glob.glob("Test_data/mydata/*.png")
nrep = 1  # repeat images
save_pos = False  # save positions
apply_otf_gt = True  # apply otf to ground truth
is_deconv = 1
num = 5
px = 1 / num
if 'hexagonal' in lattice:
    py = 2 / sqrt(3) / num
else:
    py = 1 / num
disp = np.zeros((num ** 2, 2))
for i in range(num):
    disp[i * num:(i + 1) * num, 1] = np.linspace(0, px * (num - 1), num)
    disp[i * num:(i + 1) * num, 0] = -i * py
random.seed(1234)


def main():
    # when python xx.py in console
    if len(sys.argv) == 1:
        with Pool(40) as p:
            p.map(process_image,files)
        # for n1 in range(len(files)):
        #     process_image(files[n1])


# variance will change for every image
def get_params():
    opt = argparse.Namespace()
    # system
    opt.na = 1.3 + 0.2 * (np.random.rand() - 0.5)
    # parameter of camera noise_a=QE*AD noise_b=AD^2*sigma^2 refer to
    # "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data"
    opt.noise_a = 5e-4 + 2e-4 * (np.random.rand() - 0.5)
    opt.noise_b = 0 + 0 * (np.random.rand() - 0.5)
    # lattice parameter
    opt.scale = 0.75 * (1 + 0.1 * (np.random.rand() - 0.5))  # fp is the scale times of fco
    opt.mean_intensity = 0.5  # average intensity
    opt.mod_fac = 0.8 + 0.3 * (np.random.rand() - 0.5)  # modulation factor
    opt.disp_err = 0.1 * (np.random.rand(num ** 2, 2) - 0.5)
    opt.rotate_angle = pi * (np.random.rand() - 0.5)
    # opt.rotate_angle = 0
    return opt


def psf_otf(na, is_deconv=0):
    f_rho0 = na / lambda_illu
    f_rho_z = f_rho * (f_rho <= 2 * f_rho0)
    otf = (2 / pi) * (arccos(f_rho_z / (2 * f_rho0)) - (f_rho_z / (2 * f_rho0)) * (
        sqrt(1 - (f_rho_z / (2 * f_rho0)) ** 2))) * (f_rho <= 2 * f_rho0)
    if is_deconv:
        otf = (otf > 0).astype('float')
    psf = np.abs(fftshift(ifft2(ifftshift(otf))))
    psf = psf / np.max(psf)

    return psf, otf


# generate the modulated image
def generate_sim_image(ima, opt):
    # psf
    psf, otf = psf_otf(opt.na)
    psf_gt, otf_gt = psf_otf(opt.na + opt.scale * 1.25 * lambda_illu / lambda_ex, is_deconv)
    d_ima = ima.astype('float')
    fp = opt.scale * fco
    # Illuminating pattern, ideal pattern
    if 'square' in lattice:
        a_ideal = 1/2*(cos(2*pi*fp*(cos(opt.rotate_angle+pi/4)*x+sin(opt.rotate_angle+pi/4)*y)) + cos(
            2*pi*fp*(cos(opt.rotate_angle+3*pi/4)*x+sin(opt.rotate_angle+3*pi/4)*y)))
    elif 'hexagonal' in lattice:
        a_ideal = 1/3*(cos(2 * pi * fp * (cos(opt.rotate_angle+pi/2)*x+sin(opt.rotate_angle+pi/2)*y)) + cos(
            2 * pi * fp * (cos(opt.rotate_angle+pi/6)*x+sin(opt.rotate_angle+pi/6)*y)) + cos(
            2 * pi * fp * (cos(opt.rotate_angle+5*pi/6)*x+sin(opt.rotate_angle+5*pi/6)*y)))
    elif 'line' in lattice:
        a_ideal = cos(2 * pi * fp * (cos(opt.rotate_angle)*x+sin(opt.rotate_angle)*y))
    else:
        a_ideal = 0
    p_pattern = opt.mean_intensity + opt.mod_fac * (a_ideal ** 2 - 0.5)

    # wide_image
    fd_ima = fftshift(fft2(ifftshift(d_ima)))
    f_wide_image = fd_ima * otf
    f_wide_gt = fd_ima * otf_gt
    wide_image = np.abs(fftshift(ifft2(ifftshift(f_wide_image))))
    # wide_image = (wide_image - np.min(wide_image)) / np.max(wide_image)
    wide_gt = np.abs(fftshift(ifft2(ifftshift(f_wide_gt))))

    # noise added wide_image
    sigma_noise = sqrt(opt.noise_a * wide_image + opt.noise_b)
    noise_frac = 1  # may be set to 0 to avoid noise addition
    noise_w = np.random.normal(0, 1, (nx, ny)) * sigma_noise
    n_wide_image = wide_image + noise_frac * noise_w

    images = []
    if data == 'test':
        # position, phase shift
        position = (disp + opt.disp_err) * (1 / (2 * fp))
        name1 = '%s_frequency.txt' % (opt.output_name.replace('.tif', ''))
        name2 = '%s_position.txt' % (opt.output_name.replace('.tif', ''))
        if save_pos:
            np.savetxt(name1, np.array([opt.scale * fco]), fmt='%.4e', delimiter=',')
            np.savetxt(name2, position - position[0], fmt='%.4e', delimiter=',')
            print('frequency', np.array([opt.scale * fco]))
            print('position', position - position[0])

        # pattern with positions
        fp_pattern = fftshift(fft2(ifftshift(p_pattern)))
        pattern = []
        for i1 in range(np.size(position, 0)):
            tmp_pattern = fftshift(
                ifft2(ifftshift(fp_pattern * exp(1j * 2 * pi * (position[i1, 0] * fx + position[i1, 1] * fy)))))
            pattern.append(np.real(tmp_pattern))
        # pattern = np.array(pattern)

        # modulated images
        raw_images = []

        for i2 in range(np.size(position, 0)):
            image_illu = pattern[i2] * d_ima
            fimage_illu = fftshift(fft2(ifftshift(image_illu)))
            f_raw_image = fimage_illu * otf
            raw_image_tmp = np.abs(fftshift(ifft2(ifftshift(f_raw_image))))
            raw_images.append(raw_image_tmp)  # value (>1) as 1
        raw_images = np.array(raw_images)
        # raw_images = (raw_images - np.min(raw_images)) / np.max(raw_images)

        # noise added raw_images
        for i3 in range(np.size(position, 0)):
            raw_image = raw_images[i3]
            sigma_noise = sqrt(opt.noise_a * raw_image + opt.noise_b)
            noise_r = np.random.normal(0, 1, (nx, ny)) * sigma_noise
            n_raw_image = raw_image + noise_frac * noise_r
            images.append(n_raw_image)

        images.append(n_wide_image)
        images.append(otf)
        if apply_otf_gt:
            images.append(wide_gt)
        else:
            images.append(d_ima)
        images.append(p_pattern)

    elif data == 'train':
        image_illu = p_pattern * d_ima
        fimage_illu = fftshift(fft2(ifftshift(image_illu)))
        f_raw_image = fimage_illu * otf
        raw_image = np.abs(fftshift(ifft2(ifftshift(f_raw_image))))
        sigma_noise = sqrt(opt.noise_a * raw_image + opt.noise_b)
        noise_r = np.random.normal(0, 1, (nx, ny)) * sigma_noise
        n_raw_image = raw_image + noise_frac * noise_r

        # noise added raw SIM images
        images.append(n_raw_image)
        images.append(n_wide_image)
        images.append(otf)
        if apply_otf_gt:
            images.append(wide_gt)
        else:
            images.append(d_ima)
        images.append(p_pattern)

    stack = np.array(images)

    # normalise maybe rawimage should be normalised as a whole
    for i4 in range(len(stack)-1):
        stack[i4] = (stack[i4] - np.min(stack[i4])) / (np.max(stack[i4]) - np.min(stack[i4]))

    stack = (stack * 255).astype('uint8')

    if opt.output_name is not None:
        io.imsave(opt.output_name, stack)

    return stack


def process_image(file):
    ima = io.imread(file) / 255
    ima = transform.resize(ima, (nx, ny), anti_aliasing=True)

    if len(ima.shape) > 2 and ima.shape[2] > 1:
        ima = ima.mean(2)  # if not grayscale

    filename = os.path.basename(file).replace('.png', '')

    print('Generating SIM frames for', file)

    for n5 in range(nrep):
        opt = get_params()
        opt.output_name = '%s/%s_%d.tif' % (out_dir, filename, n5)
        generate_sim_image(ima, opt)


# run this file
if __name__ == '__main__':
    main()
