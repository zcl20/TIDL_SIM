# import math
import os
import torch
import torch.nn as nn
import time
# import numpy as np
import torch.optim as optim
# import torchvision
# from torch.autograd import Variable
# import subprocess
from Models import get_model
from Data_Handler import get_data_loader
from Plotting import test_combined_plots, generate_convergence_plots
from torch.utils.tensorboard import SummaryWriter
# from options import parser
import sys
# import glob


def train(config, fid, data_loader, valid_loader, net):
    checkpoint = []
    writer = []
    start_epoch = 0
    # define loss function
    if config['TRAIN']['LOSSES'] == 'MSELoss':
        loss_function = nn.MSELoss()
        # loss_function = SSIM(data_range=255)
        # loss_function = VGGLoss()
    else:
        loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config['TRAIN']['LR'])
    device = torch.device('cuda' if torch.cuda.is_available() and not config['TEST']['CPU'] else 'cpu')
    loss_function.cuda()
    if len(config['TRAIN']['WEIGHTS_DIR']) > 0:  # load previous weights?
        checkpoint = torch.load(config['TRAIN']['WEIGHTS_DIR'])
        print('loading checkpoint', config['TRAIN']['WEIGHTS_DIR'])

        net.load_state_dict(checkpoint['state_dict'])
        if config['TRAIN']['LR'] == 1:  # continue as it was
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    step_size, gamma = int(config['TRAIN']['SCHEDULER'].split(
            ',')[0]), float(config['TRAIN']['SCHEDULER'].split(',')[1])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
    if len(config['TRAIN']['WEIGHTS_DIR']) > 0:
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

    t0 = time.perf_counter()

    for epoch in range(start_epoch, config['TRAIN']['EPOCHS']):
        count = 0
        mean_loss = 0

        # for param_group in optimizer.param_groups:
        #     print('\nLearning rate', param_group['lr'])

        for i, bat in enumerate(data_loader):
            lr, hr = bat[0], bat[1]
            optimizer.zero_grad()

            sr = net(lr.to(device))
            loss = loss_function(sr, hr.to(device))
            # loss = loss_function(sr.repeat(1, 3, 1, 1), (hr.to(opt.device)).repeat(1, 3, 1, 1))

            loss.backward()
            optimizer.step()

            # Status and display
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch + 1, config['TRAIN']['EPOCHS'],
                                                    i + 1, len(data_loader), loss.data.item()), end='')

            count += 1
            if config['TRAIN']['LOG'] and count * config['TRAIN']['BATCH_SIZE'] // 1000 > 0:
                t1 = time.perf_counter() - t0
                mem = torch.cuda.memory_allocated()
                writer = SummaryWriter(log_dir=config['TRAIN']['OUTPUT_DIR'], comment='_%s_%s' % (
                    config['TRAIN']['OUTPUT_DIR'].replace('\\', '/').split('/')[-1], config['MODEL']['NAME']))
                writer.add_scalar(
                    'data/mean_loss_per_1000', mean_loss / count, epoch)
                writer.add_scalar('data/time_per_1000', t1, epoch)
                train_stats = open(config['TRAIN']['OUTPUT_DIR'].replace(
                    '\\', '/') + '/train_stats.csv', 'w')
                print(epoch, count * config['TRAIN']['BATCH_SIZE'], t1, mem,
                      mean_loss / count, file=train_stats)
                train_stats.flush()
                count = 0

        # Scheduler
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('\nLearning rate', param_group['lr'])
            break

        # Printing
        mean_loss = mean_loss / len(data_loader)
        t1 = time.perf_counter() - t0
        eta = (config['TRAIN']['EPOCHS'] - (epoch + 1)) * t1 / (epoch + 1)
        o_str = '\nEpoch [%d/%d] done, mean loss: %0.6f, time spent: %0.1fs, ETA: %0.1fs' % (
            epoch + 1, config['TRAIN']['EPOCHS'], mean_loss, t1, eta)
        print(o_str)
        print(o_str, file=fid)
        fid.flush()
        if config['TRAIN']['LOG']:
            writer.add_scalar(
                'data/mean_loss', mean_loss / len(data_loader), epoch)

        # TEST
        if (epoch + 1) % config['TEST']['TEST_INTERVAL'] == 0:
            test_combined_plots(net, valid_loader, config, fid, epoch)

        if (epoch + 1) % config['TRAIN']['SAVE_INTERVAL'] == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, '%s/prelim%d.pth' % (config['TRAIN']['OUTPUT_DIR'], epoch + 1))

    checkpoint = {'epoch': config['TRAIN']['EPOCHS'], 'state_dict': net.state_dict(),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, config['TRAIN']['OUTPUT_DIR'] + '/final.pth')


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return torch.mean(torch.pow((x - y), 2))


def main(config):

    os.makedirs(config['TRAIN']['OUTPUT_DIR'], exist_ok=True)
    fid = open(config['TRAIN']['OUTPUT_DIR'] + '/log.txt', 'w')

    o_str = 'ARGS: ' + ' '.join(sys.argv[:])
    print(config, '\n')
    print(config, '\n', file=fid)
    print('\n%s\n' % o_str)
    print('\n%s\n' % o_str, file=fid)

    print('getting dataloader', config['TRAIN']['TRAIN_DATA_DIR'])
    data_loader, valid_loader = get_data_loader(config)

    if config['TRAIN']['LOG']:
        train_stats = open(config['TRAIN']['OUTPUT_DIR'].replace(
            '\\', '/') + '/train_stats.csv', 'w')
        test_stats = open(config['TRAIN']['OUTPUT_DIR'].replace(
            '\\', '/') + '/test_stats.csv', 'w')
        print('iter,n_sample,time,memory,mean_loss', file=train_stats)
        print('iter,time,memory,psnr,ssim', file=test_stats)

    t0 = time.perf_counter()
    config['TEST']['T0'] = t0
    net = get_model(config)

    if not config['TEST']['STATUE']:
        train(config, fid, data_loader, valid_loader, net)
        # torch.save(net.state_dict(), opt.out + '/final.pth')
    else:
        if len(config['TRAIN']['WEIGHTS_DIR']) > 0:  # load previous weights?
            checkpoint = torch.load(config['TRAIN']['WEIGHTS_DIR'])
            print('loading checkpoint', config['TRAIN']['WEIGHTS_DIR'])
            net.load_state_dict(checkpoint['state_dict'])
            print('time: %0.1f' % (time.perf_counter() - t0))
        test_combined_plots(net, valid_loader, config, fid)

    fid.close()
    if not config['TEST']['STATUE']:
        generate_convergence_plots(config, config['TRAIN']['OUTPUT_DIR'] + '/log.txt')

    print('time: %0.1f' % (time.perf_counter() - t0))
