import torch
import matplotlib.pyplot as plt
# import torchvision
# import skimage
from skimage.metrics import structural_similarity
# from skimage.measure import compare_ssim
import torchvision.transforms as transforms
import numpy as np
import time
# from PIL import Image

# import scipy.ndimage as ndimage
# import torch.nn as nn
# import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.switch_backend('agg')

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()


def test_combined_plots(net, loader, config, fid, idx=0):
    def psnr_numpy(p0, p1):
        i0, i1 = np.array(p0) / 255.0, np.array(p1) / 255.0
        mse = np.mean((i0 - i1) ** 2)
        psnr = 20 * np.log10(1 / np.sqrt(mse))
        return psnr

    def ssim_numpy(p0, p1):
        i0, i1 = np.array(p0) / 255.0, np.array(p1) / 255.0
        return structural_similarity(i0, i1, data_range=1)

    def calc_scores(img, hr=None, make_plot=False, plot_idx=0, title=None):
        if make_plot:
            plt.subplot(1, 3, plot_idx)
            plt.gca().axis('off')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img, cmap='gray')
        if hr is None:
            if make_plot:
                plt.title(r'GT ($\infty$/1.000)')
        else:
            psnr, ssim = psnr_numpy(img, hr), ssim_numpy(img, hr)
            if make_plot:
                plt.title('%s (%0.2fdB/%0.3f)' % (title, psnr, ssim))
            return psnr, ssim

    count, mean_bc_psnr, mean_sr_psnr, mean_bc_ssim, mean_sr_ssim = 0, 0, 0, 0, 0

    for i, bat in enumerate(loader):
        # lr(input lower) hr(gt) sr(output from net)
        lr_bat, hr_bat, wf_bat = bat[0], bat[1], bat[2]
        with torch.no_grad():
            sr_bat = net(lr_bat.to(device))
        sr_bat = sr_bat.cpu()

        for j in range(len(lr_bat)):  # loop over batch
            make_plot = ((idx < 1 or (
                    idx + 1) % config['TEST']['PLOT_INTERVAL'] == 0 or idx == config['TRAIN']['EPOCHS'] - 1)
                         and count < config['TEST']['NUM_PLOT'])
            if config['TEST']['STATUE']:
                make_plot = True

            wf, sr, hr = wf_bat.data[j], sr_bat.data[j], hr_bat.data[j]

            sr = torch.clamp(sr, min=0, max=1)

            # fix to deal with 3D deconvolution
            if config['MODEL']['OUT_CHANNELS'] > 1:
                wf = wf[wf.shape[0] // 2]  # channels are not for colours but separate grayscale frames, take middle
                sr = sr[sr.shape[0] // 2]
                hr = hr[hr.shape[0] // 2]

            # Common commands
            # lr, bc, sr, hr = toPIL(lr), toPIL(bc), toPIL(sr), toPIL(hr)
            wf, sr, hr = toPIL(wf), toPIL(sr), toPIL(hr)

            if make_plot:
                plt.figure(figsize=(10, 5), facecolor='white')
            bc_psnr, bc_ssim = calc_scores(wf, hr, make_plot, plot_idx=1, title='WF')
            sr_psnr, sr_ssim = calc_scores(sr, hr, make_plot, plot_idx=2, title='SR')
            calc_scores(hr, None, make_plot, plot_idx=3)

            mean_bc_psnr += bc_psnr
            mean_sr_psnr += sr_psnr
            mean_bc_ssim += bc_ssim
            mean_sr_ssim += sr_ssim

            if make_plot:
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                plt.savefig('%s/combined_epoch%d_%d.png' % (config['TRAIN']['OUTPUT_DIR'], idx + 1, count),
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
                if config['TEST']['STATUE']:
                    wf.save('%s/lr_epoch%d_%d.png' % (config['TRAIN']['OUTPUT_DIR'], idx + 1, count))
                    sr.save('%s/sr_epoch%d_%d.png' % (config['TRAIN']['OUTPUT_DIR'], idx + 1, count))
                    hr.save('%s/hr_epoch%d_%d.png' % (config['TRAIN']['OUTPUT_DIR'], idx + 1, count))

            count += 1
            if count == config['TEST']['NUM_TEST']:
                break
        if count == config['TEST']['NUM_TEST']:
            break

    summary_str = ""
    if count == 0:
        summary_str += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summary_str += 'Testing of %d samples complete. bc: %0.2f dB / %0.4f, sr: %0.2f dB / %0.4f \n' % (
        count, mean_bc_psnr / count, mean_bc_ssim / count, mean_sr_psnr / count, mean_sr_ssim / count)
    print(summary_str)
    print(summary_str, file=fid)
    fid.flush()
    if config['TRAIN']['LOG'] and not config['TEST']['STATUE']:
        t1 = time.perf_counter() - config['TEST']['T0']
        mem = torch.cuda.memory_allocated()
        test_stats = open(config['TRAIN']['OUTPUT_DIR'].replace(
            '\\', '/') + '/test_stats.csv', 'w')
        print(idx, t1, mem, mean_sr_psnr / count, mean_sr_ssim / count, file=test_stats)
        test_stats.flush()


def generate_convergence_plots(config, filename):
    fid = open(filename, 'r')
    psnr_list = []
    ssim_list = []

    for line in fid:
        if 'sr: ' in line:
            psnr_list.append(float(line.split('sr: ')[1].split(' dB')[0]))
            ssim_list.append(float(line.split('sr: ')[1].split(' / ')[1]))

    plt.figure(figsize=(12, 5), facecolor='white')
    plt.subplot(121)
    plt.plot(psnr_list, '.-')
    plt.title('PSNR')

    plt.subplot(122)
    plt.plot(ssim_list, '.-')
    plt.title('SSIM')

    plt.savefig('%s/convergencePlot.png' % config['TRAIN']['OUTPUT_DIR'], dpi=300)
