# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
import collections

# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area

def show_performance(distortion_name,
                     net,
                     alexnet,
                     imagenet_clean_path,
                     imagenet_c_path,
                     mean, std,
                     batch_size):
    errs_resnet = []
    errs_alexnet = []
    n = 0
    with torch.no_grad():

        for severity in range(1, 6):
            curr_severity_path = os.path.join(imagenet_c_path, distortion_name, str(severity))
            if os.path.exists(curr_severity_path):
                n += 1
                distorted_dataset = dset.ImageFolder(
                    root=curr_severity_path,
                    transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))

                distorted_dataset_loader = torch.utils.data.DataLoader(
                    distorted_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

                correct_resnet = 0
                correct_alexnet = 0
                for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                    data = data.cuda()

                    output_resnet = net(data)
                    pred_resnet = output_resnet.data.max(1)[1]
                    correct_resnet += pred_resnet.eq(target.cuda()).sum()

                    output_alexnet = alexnet(data)
                    pred_alexnet = output_alexnet.data.max(1)[1]
                    correct_alexnet += pred_alexnet.eq(target.cuda()).sum()

                errs_resnet.append(1 - 1.*correct_resnet / len(distorted_dataset))
                errs_alexnet.append(1 - 1.*correct_alexnet / len(distorted_dataset))
        print('\t(n={}) Imagenet-c ResNet18 Errors: {}'.format(n, tuple(errs_resnet)), flush=True)
        print('\t(n={}) Imagenet-c AlexNet Errors: {}'.format(n, tuple(errs_alexnet)), flush=True)

        correct_resnet = 0
        correct_alexnet = 0
        for batch_idx, (data, target) in enumerate(clean_loader):
            data = data.cuda()

            output_resnet = net(data)
            pred_resnet = output_resnet.data.max(1)[1]
            correct_resnet += pred_resnet.eq(target.cuda()).sum()

            output_alexnet = net(data)
            pred_alexnet = output_alexnet.data.max(1)[1]
            correct_alexnet += pred_alexnet.eq(target.cuda()).sum()

        clean_error_resnet = 1 - correct_resnet / len(clean_loader.dataset)
        clean_error_alexnet = 1 - correct_alexnet / len(clean_loader.dataset)
        print('\tImagenet Clean ResNet18 Errors: {}'.format(clean_error_resnet), flush=True)
        print('\tImagenet Clean AlexNet Errors: {}'.format(clean_error_alexnet), flush=True)

    ce_unnormalized = torch.mean(errs_resnet).detach().cpu().numpy()
    ce_normalized = (torch.sum(errs_resnet) / torch.sum(errs_alexnet)).detach().cpu().numpy()
    relative_ce = ((torch.sum(errs_resnet) - clean_error_resnet) / (torch.sum(errs_alexnet) - clean_error_alexnet)).detach().cpu().numpy()
    return ce_unnormalized, ce_normalized, relative_ce


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////

def eval_model(net, batch_size=256, seed=0):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    net.cuda()
    net.eval()
    alexnet = models.alexnet(pretrained=True)
    alexnet.cuda()
    alexnet.eval()

    cudnn.benchmark = True

    imagenet_clean_path = "/scratch/ssd002/datasets/imagenet/val"
    imagenet_c_path = "/scratch/hdd001/home/slowe/imagenet-c"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
        root=imagenet_clean_path,
        transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
        batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    print('\nUsing ImageNet data')

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

    errors_ce_unnormalized = []
    errors_ce_normalized = []
    errors_relative_ce = []
    for distortion_name in distortions:
        curr_dist_path = os.path.join(imagenet_c_path, distortion_name)
        if os.path.exists(curr_dist_path):
            print('======== Distortion: {:15s}'.format(distortion_name), flush=True)
            ce_unnormalized, ce_normalized, relative_ce = show_performance(distortion_name,
                                                                           net,
                                                                           alexnet,
                                                                           imagenet_clean_path,
                                                                           imagenet_c_path,
                                                                           mean, std,
                                                                           batch_size)
            errors_ce_unnormalized.append(ce_unnormalized)
            errors_ce_normalized.append(ce_normalized)
            errors_relative_ce.append(relative_ce)
            print('\tCE (unnormalized) (%): {:.2f}  |  CE (normalized) (%): {:.2f}  |  Relative CE (%): {:.2f}\n'.format(
                100 * ce_unnormalized, 100 * ce_normalized, 100 * relative_ce), flush=True)

    print('\nmCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(errors_ce_unnormalized)), flush=True)
    print('mCE (normalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(errors_ce_normalized)), flush=True)
    print('Relative mCE (%): {:.2f}'.format(100 * np.mean(errors_relative_ce)), flush=True)

if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    eval_model(net)
