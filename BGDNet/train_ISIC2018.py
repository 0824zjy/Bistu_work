# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
from datetime import datetime
import logging
import time
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# 新模型（models/MSDUNet.py 中提供）
from models.MSDUNet import MSDUFormer_Xnet_CrossSwin_ECH

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

from ptflops import get_model_complexity_info
warnings.filterwarnings("ignore")


def structure_loss(pred, mask):
    """
    与你原始的一致：吃 logits (B,1,H,W)，内部做 BCEWithLogits + 加权IoU
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


@torch.no_grad()
def test(model, data_path, dataset):
    """
    适配新模型输出：
      - 优先使用 out["fused_prob"]（概率）；若无，则使用 logits -> sigmoid
    其它评估流程保持不变
    """
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    test_loader = test_dataset(image_root, gt_root, args.img_size)
    num1 = test_loader.size

    DSC = 0.0
    for _ in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        out = model(image)

        # 兼容：新模型返回 dict；旧模型直接返回 tensor
        if isinstance(out, dict):
            if 'fused_prob' in out:
                res = out['fused_prob']  # (B,1,h,w) 概率
            elif 'pixel_prob' in out:
                res = out['pixel_prob']  # (B,1,h,w)
            elif 'logits' in out:
                res = torch.sigmoid(out['logits'])
            else:
                # 最保险的回退
                keys = list(out.keys())
                raise RuntimeError(f"Unexpected model outputs: {keys}")
        else:
            res = torch.sigmoid(out)  # 旧模型：logits

        # 上采样到 GT 尺寸；注意 res 维度 (B,1,h,w)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # Dice
        pred_bin = (res >= 0.5)
        target = (gt >= 0.5)
        smooth = 1
        intersection = (pred_bin.astype(np.float32) * target.astype(np.float32)).sum()
        dice = (2 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)
        DSC += float(dice)

    return DSC / num1, num1


def train(train_loader, model, optimizer, epoch, test_path, model_name='SUFormer'):
    model.train()
    global best
    global total_train_time
    global max_memory_usage
    max_memory_usage = 0
    time_before_epoch_start = time.time()
    size_rates = [1]
    loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(args.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            outputs = model(images)

            # ✅ 兼容：新模型返回 dict，取 logits 以保持原损失不变
            if isinstance(outputs, dict):
                logits = outputs.get('logits', None)
                if logits is None:
                    raise RuntimeError("Model must return 'logits' in training for structure_loss.")
            else:
                logits = outputs

            loss = structure_loss(logits, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss, args.batchsize)

            # ---- monitoring maximum memory usage ----
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated()
                max_memory_usage = max(max_memory_usage, current_memory)

        # ---- train visualization ----
        if i % 50 == 0 or i == total_step:
            # 将 show() 的返回安全转为 float 再格式化
            loss_val = float(loss_record.show())
            inter_info = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}, max memory usage: {:.2f} MB'.format(
                datetime.now(), epoch, args.epoch, i, total_step, loss_val,
               (max_memory_usage / (1024 ** 2)) if torch.cuda.is_available() else 0.0
            )
            print(inter_info)
            logging.info(inter_info)

    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: ' + str(total_train_time))
    logging.info('total train time till current epoch: ' + str(total_train_time))

    # save model
    save_path = (args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}-last.pth'))

    # choose the best model
    global dict_plot

    if (epoch + 1) % 1 == 0:
        total_dice = 0
        total_images = 0
        for dataset in ['test']:
            dataset_dice, n_images = test(model, test_path, dataset)
            total_dice += (n_images * dataset_dice)
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)

        dataset_test_dice = total_dice / total_images
        meandice = dataset_test_dice
        valid_meandice = dataset_test_dice
        dict_plot['valid'].append(valid_meandice)
        dict_plot['test'].append(dataset_test_dice)
        print('Test dice score: {}'.format(dataset_test_dice))
        logging.info('Test dice score: {}'.format(dataset_test_dice))

        if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}-best.pth'))

    # reset cuda peak mem
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    dict_plot = {'valid': [], 'test': []}
    name = ['valid', 'test']

    ################## model_name #############################
    model_name = 'ISIC2018_ECH_CrossSwinTransX'

    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=str, default='SUFormer', help='Name of encoder: PVT or MERIT')
    parser.add_argument("--num_classes", type=int, default=1, help="output channel of network")
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation', default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
    parser.add_argument('--img_size', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2_Training_Input/',
                        help='path to train dataset')
    parser.add_argument('--test_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2/test/',
                        help='path to testing Kvasir dataset')
    parser.add_argument('--train_save', type=str, default='./model_out/ISIC2018-5/' + model_name + '/')

    args = parser.parse_args()

    if not os.path.exists(args.train_save):
        os.makedirs(args.train_save)

    logging.basicConfig(filename=os.path.join(args.train_save, 'train.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build model ----
    model = MSDUFormer_Xnet_CrossSwin_ECH(num_classes=args.num_classes).cuda(0)
    logging.info(model)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    best = 0
    params = model.parameters()

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    # ---- data ----
    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize, trainsize=args.img_size,
                              shuffle=True, augmentation=args.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0

    for epoch in range(1, args.epoch):
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        train(train_loader, model, optimizer, epoch, args.test_path, model_name=model_name)

    print('avg train time: ' + str(total_train_time / (args.epoch - 1)))
    logging.info('avg train time: ' + str(total_train_time / (args.epoch - 1)))
