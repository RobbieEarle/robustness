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

parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Architecture
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////

net = models.resnet18(pretrained=True)

args.prefetch = 4

for p in net.parameters():
    p.volatile = True

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(1)
np.random.seed(1)
if args.ngpu > 0:
    torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

args.test_bs = 256
print('Model Loaded')

# /////////////// Data Loader ///////////////

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
    root="/scratch/ssd002/datasets/imagenet/val",
    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
    batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


# correct = 0
# for batch_idx, (data, target) in enumerate(clean_loader):
#     data = V(data.cuda(), volatile=True)
#
#     output = net(data)
#
#     pred = output.data.max(1)[1]
#     correct += pred.eq(target.cuda()).sum()
#
# clean_error = 1 - correct / len(clean_loader.dataset)
# print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):

    errs_resnet = []
    errs_alexnet = []
    n = 0
    for severity in range(1, 6):
        if os.path.exists('/scratch/ssd002/datasets/imagenet-c/' + distortion_name + '/' + str(severity)):
            n += 1
            distorted_dataset = dset.ImageFolder(
                root='/scratch/ssd002/datasets/imagenet-c/' + distortion_name + '/' + str(severity),
                transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))

            distorted_dataset_loader = torch.utils.data.DataLoader(
                distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

            correct = 0
            for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                data = V(data.cuda(), volatile=True)

                output_resnet = net(data)
                pred_resnet = output_resnet.data.max(1)[1]
                correct_resnet += pred_resnet.eq(target.cuda()).sum()

                output_alexnet = alexnet(data)
                pred_alexnet = output_alexnet.data.max(1)[1]
                correct_alexnet += pred_alexnet.eq(target.cuda()).sum()

            errs_resnet.append(1 - 1.*correct_resnet / len(distorted_dataset))
            errs_alexnet.append(1 - 1.*correct_alexnet / len(distorted_dataset))
    print('\t(n={}) Imagenet-c ResNet18 Errors: {}'.format(tuple(errs_resnet)))
    print('\t(n={}) Imagenet-c AlexNet Errors: {}'.format(tuple(errs_alexnet)))

    correct = 0
    for batch_idx, (data, target) in enumerate(clean_loader):
        data = V(data.cuda(), volatile=True)

        output_resnet = net(data)
        pred_resnet = output_resnet.data.max(1)[1]
        correct_resnet += pred_resnet.eq(target.cuda()).sum()

        output_alexnet = net(data)
        pred_alexnet = output_alexnet.data.max(1)[1]
        correct_alexnet += pred_alexnet.eq(target.cuda()).sum()

    clean_error_resnet = 1 - correct_resnet / len(clean_loader.dataset)
    clean_error_alexnet = 1 - correct_alexnet / len(clean_loader.dataset)
    print('\tImagenet Clean ResNet18 Errors:', clean_error_resnet)
    print('\tImagenet Clean AlexNet Errors:', clean_error_alexnet)


    ce_unnormalized = np.mean(errs_resnet)
    ce_normalized = np.sum(errs_resnet) / np.sum(errs_alexnet)
    relative_ce = (np.sum(errs_resnet) - clean_error_resnet) / (np.sum(errs_alexnet) - clean_error_alexnet)
    return ce_unnormalized, ce_normalized, relative_ce


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

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
alexnet = models.alexnet(pretrained=True)
for distortion_name in distortions:
    if os.path.exists('/scratch/ssd002/datasets/imagenet-c/' + distortion_name):
        print('Distortion: {:15s}'.format(distortion_name))
        ce_unnormalized, ce_normalized, relative_ce = show_performance(distortion_name)
        errors_ce_unnormalized.append(ce_unnormalized)
        errors_ce_normalized.append(ce_normalized)
        errors_relative_ce.append(relative_ce)
        print('\tCE (unnormalized) (%): {:.2f}  |  CE (normalized) (%): {:.2f}  |  Relative CE (%): {:.2f}'.format(
            100 * ce_unnormalized, 100 * ce_normalized, 100 * relative_ce))

print('\nmCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(errors_ce_unnormalized)))
print('mCE (normalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(errors_ce_normalized)))
print('Relative mCE (%): {:.2f}'.format(100 * np.mean(errors_relative_ce)))
