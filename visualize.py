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
import datasets.dataset as dataset
import math
import time
import json
from PIL import Image

from datasets.image import *
from models.unet import U_Net, U_Net_light
from utils.inference_flux import trans_n8, flux_to_skl
from utils.inference_from_direction_map import direction_to_cells_new
from metrics.PR_Hungarian import cal_pr
from metrics.Utils_Hungarian import generate_pred_gt_loc

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--model_id', default=0, type=int, help='id of trained model')
parser.add_argument('--gpu_id', default=0, type=int, help='id of trained model')

parser.add_argument('--task', default="saved_model/UNet/train50_unet_new/", type=str, help='id of trained model')
parser.add_argument('--infer_thresh', default=0.6, type=float, help='id of trained model')
parser.add_argument('--distance_thresh', default=6, type=int, help='id of trained model')

parser.add_argument('--batch_size', default=1, type=int, help='id of trained model')
parser.add_argument('--workers', default=4, type=int, help='id of trained model')
parser.add_argument('--print_freq', default=200, type=int, help='id of trained model')
parser.add_argument('--seed', default=200, type=int, help='id of trained model')
warnings.filterwarnings('ignore')

def main():

    best_prec1 = 1e6
    args = parser.parse_args()
    args.seed = time.time()
    args.result = os.path.join(args.task, 'vis')
    args.pr_dir = os.path.join(args.task, 'pr_eval')
    eval_file = os.path.join(args.pr_dir, 'eval.json')
    eval_txt = os.path.join(args.pr_dir, 'eval.txt')
    pre = os.path.join(args.task, 'model_best.pth.tar')

    # visualize predicted result
    if not os.path.exists(args.result):
        os.mkdir(args.result)

    # evaluate localization task
    if not os.path.exists(args.pr_dir):
        os.mkdir(args.pr_dir)

    # load test data
    with open('./Cellstest.npy', 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    print(len(val_list), val_list[0],len(val_list))

    model = U_Net()
    model = nn.DataParallel(model, device_ids=[args.gpu_id])
    model = model.cuda('cuda:{}'.format(args.gpu_id))

    if pre:
        if os.path.isfile(pre):
            print("=> loading checkpoint '{}'".format(pre))
            checkpoint = torch.load(pre)
            best_prec1 = checkpoint['best_prec1']

            model_dict = model.state_dict()
            pre_val = checkpoint['state_dict']
            # pre_val = {k: v for k, v in pre_val.items() if k in model_dict}
            pre_val = {k: v for k, v in pre_val.items() if 'backbone' in k}
            model_dict.update(pre_val)
            model.load_state_dict(model_dict)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(pre))

    with torch.no_grad():
        mae, precision, recall, f1_score = validate(val_list, model, args)

        eval_dict = {'Thresh':args.infer_thresh,'distance':args.distance_thresh,'MAE':mae, 'Precision':100.0*precision, \
                     'Recall':100.0*recall, 'F1_Score':100.0*f1_score}

        with open(eval_file, 'w') as f:
            json_str = json.dumps(eval_dict, indent=4, ensure_ascii=False)
            f.write(json_str)

        wstr = 'Thresh:{}, Distance:{} ,MAE:{}, Precision:{:.3f}, Recall:{:.3f}, F1_score:{:.3f}\n'.format(args.infer_thresh,\
            args.distance_thresh,mae, 100.0*precision, 100.0*recall, 100.0*f1_score)
        with open(eval_txt, 'a+') as f:
            f.write(wstr)

        print(wstr)



def count_distance(input_image, input_pred, gt, fname, args):
    gt = gt.numpy()
    gt = np.squeeze(gt)
    gt_loc = np.array(list(np.where(gt==1))).transpose((1,0))

    mask, output = direction_to_cells_new(input_pred, args.infer_thresh)
    # mask, output = flux_to_skl(input_pred, args.infer_thresh)
    output = (output>0).astype(np.uint8)
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    center = []
    for j, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        center_x = int(x + w // 2)
        center_y = int(y + h // 2)
        center.append([center_y, center_x])
    count = len(contours)
    gt_center = np.argwhere(gt)
    center = np.array(center)

    return output, center, gt_center


def validate(Pre_data, model, args):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset_CRC_Val(Pre_data, args.task, shuffle=False,
            transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]), ]), train=False),batch_size=1)

    model.eval()

    mae = []
    mse = []
    error = []
    precisions = []
    recalls = []
    ids = list(range(1, len(test_loader)+1))
    pred_file = os.path.join(args.pr_dir, 'pred_loc.txt')
    gt_file = os.path.join(args.pr_dir, 'gt_loc.txt')

    for i, (img, target, k, fname, img_raw) in enumerate(test_loader):
        print(fname)
        pre_count = 0
        img = img.cuda('cuda:{}'.format(args.gpu_id))
        target = target.cuda('cuda:{}'.format(args.gpu_id))
        result = model(img, target)[0]
        original_distance_map = result.detach().cpu().numpy()
        original_distance_map = original_distance_map.squeeze(0)

        img_raw = img_raw.detach().cpu().numpy()
        img_raw = img_raw.squeeze(0)
        k = k.squeeze(0)
        Gt_count = k.sum()
        with h5py.File(os.path.join(args.result, fname[0]), 'w') as f:
            f['flux'] = original_distance_map
            f['gt'] = k
            f['img'] = img_raw

        record_pred = open(pred_file, 'a+')
        record_gt = open(gt_file, 'a+')

        component, pred_point, gt_point = count_distance(img_raw, original_distance_map, k, fname, args)
        generate_pred_gt_loc(pred_point, gt_point, record_pred, record_gt, args.distance_thresh, i+1, img_raw)

        mae.append(abs(pre_count - Gt_count))
        # error.append(pre_count - Gt_count)
        mse.append(abs(pre_count - Gt_count) * abs(pre_count - Gt_count))

    mae, precision, recall, f1_score = cal_pr(pred_file, gt_file, ids)
    return mae, precision, recall, f1_score


if __name__ == '__main__':
    main()
