# coding:utf-8
import sys
import os
import warnings

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse
import cv2
import math
import time
from PIL import Image

import datasets.dataset as dataset
from datasets.image import *
from models.unet import U_Net, U_Net_light
from models.p2pnet import *
from utils.inference_flux import trans_n8, flux_to_skl
from utils.inference_from_direction_map import direction_to_cells_new
from utils.utils import save_checkpoint, AverageMeter
from utils.misc import *

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--model_id', default=0, type=int, help='id of trained model')
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--task', default="saved_model/UNet/train50_p2pnet_new/", type=str)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--decay', default=5e-4, type=float)
parser.add_argument('--infer_thresh', default=0.6, type=float)
parser.add_argument('--row', default=1, type=int)
parser.add_argument('--line', default=1, type=int)
parser.add_argument('--eos_coef', default=0.5, type=float)
parser.add_argument('--point_loss_coef', default=0.0002, type=float)
parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--epochs', default=20000, type=int)
parser.add_argument('--print_freq', default=200, type=int)
parser.add_argument('--seed', default=200, type=int)

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
   
    best_prec1 = 1e6
    args = parser.parse_args()
    args.seed = time.time()
    if not os.path.exists(args.task):
        os.mkdir(args.task)

    with open('./Cellstrain.npy', 'rb') as outfile:
        train_list = np.load(outfile).tolist()

    with open('./Cellsval.npy', 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    print(len(train_list), train_list[0],len(val_list))

    # model = U_Net()
    device = torch.device('cuda:{}'.format(args.gpu_id))
    model, criterion_point = build(args, device, training=True)
    # model = nn.DataParallel(model, device_ids=[args.gpu_id])
    model.to(device)
    criterion = nn.MSELoss(reduce=False)
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.decay)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, model, criterion, criterion_point, optimizer, epoch, args.task, args.lr, args, device)
        with torch.no_grad():
            prec1 = validate(val_list, model, args.task, epoch, args, device)
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
            save_checkpoint({ 'epoch': epoch + 1,
                'state_dict': model.state_dict(), 'best_prec1': best_prec1,
                'presume':os.path.join(args.task, 'model_best.pth.tar'),
                'optimizer': optimizer.state_dict()}, is_best, args.task, args.model_id)


def count_distance(input_pred, input_img, thr, kpoints, fname):
    # mask, output = direction_to_cells_new(input_pred, thr)
    mask, output = flux_to_skl(input_pred, thr)
    output = (output > 0).astype(np.uint8)
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = len(contours)

    return count, output, mask


def nl(input, target, mask, criterion):
    n,c,h,w = input.size()
    # print(input.size())
    mask = np.array(mask)
    regionPos = (mask>0)
    regionNeg = (mask==0)
    sumPos = np.sum(regionPos)
    sumNeg = np.sum(regionNeg)
    weightPos = np.zeros((c, n, h, w))
    weightNeg = np.zeros((c, n, h, w))
    weight = np.zeros((c, n, h, w))
    weightPos[0] = sumNeg/float(sumPos+sumNeg)*regionPos
    weightPos[1] = sumNeg/float(sumPos+sumNeg)*regionPos
    weightNeg[0] = sumPos/float(sumPos+sumNeg)*regionNeg
    weightNeg[1] = sumPos/float(sumPos+sumNeg)*regionNeg
    weightNeg = weightNeg.transpose((1, 0, 2, 3))
    weightPos = weightPos.transpose((1, 0, 2, 3))
    weight = np.add(weightNeg, weightPos)
    weight = torch.from_numpy(weight).type(torch.FloatTensor).cuda()


    loss = (criterion(input,target)*weight).mean()
    # loss = criterion(input,target)
    return loss


def train(Pre_data, model, criterion, criterion_point, optimizer, epoch, task_id, lr, args, device):
    losses = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset_CRC(Pre_data, task_id,
                shuffle=True, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),]), train=True,
                batch_size=args.batch_size, num_workers=args.workers),
        batch_size=args.batch_size,drop_last=False)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), lr))

    model.train()

    for i, (img, target, k, fname, mask, target_points, img_raw) in enumerate(train_loader):

        loss = 0
        img = img.to(device)
        target = target.type(torch.FloatTensor).to(device)
        #UNet
        results = model(img, target, device)
        target_points[0]['point'] = target_points[0]['point'].to(device)
        target_points[0]['labels'] = target_points[0]['labels'].to(device)
        loss += nl(results[0], target, mask, criterion)
        # calc the losses
        loss_dict = criterion_point(results[1], target_points, fname, img_raw[0], img_raw[0].shape)
        weight_dict = criterion_point.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # print('losses:',losses)

        # reduce all losses
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # print('losses:',loss_dict.values())

        loss_value = losses_reduced_scaled.item()
        loss += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        # losses.update(loss.item(), img.size(0))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if i % args.print_freq == 0:
        #     print('4_Epoch: [{0}][{1}/{2}]\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #         .format(epoch, i, len(train_loader), loss=losses))


def validate(Pre_data, model, task_id, epoch, args, device):
    print('begin test')

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset_CRC_Val(Pre_data, task_id,shuffle=False,
                transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]), ]), train=False),batch_size=1)

    # unet_model = model.get_backbone()
    model.eval()
    mae = 0
    mse = 0
    visi = []

    for i, (img, target, k, fname, img_raw) in enumerate(test_loader):
        img = img.to(device)
        #UNet
        result = model(img, target, device)[0]
        original_distance_map = result.detach().cpu().numpy()
        show_map = original_distance_map.squeeze(0)
        pre_count, skl, mask = count_distance(show_map, img_raw.detach().cpu().numpy()[0], args.infer_thresh, k, fname)
        Gt_count = torch.sum(k).item()
        # print('GT_count: {}, Pred count: {}'.format(Gt_count, pre_count))

        mae += abs(pre_count - Gt_count)
        mse += abs(pre_count - Gt_count) * abs(pre_count - Gt_count)
        mask = 255.0*mask/mask.max()
        cv2.imwrite(os.path.join(task_id, fname[0].replace('.h5', '_Mask.png')), mask)

    mae = mae / len(test_loader)
    mse = math.sqrt(mse/len(test_loader))
    print('Epoch:{}:MAE:{}'.format(epoch, mae))

    return mae


if __name__ == '__main__':
    main()
