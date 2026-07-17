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

from BGDNet.models.BGDNet_or import BGDNet   # 模型
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

warnings.filterwarnings("ignore")


def dice_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + eps
    dice = 1.0 - (num / den)
    return dice.mean()


def joint_loss(pred_mask, gt_mask, pred_boundary, gt_boundary, alpha=1.0, beta=0.4):
    # L_seg = BCE + Dice
    bce_seg = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    dice = dice_loss_with_logits(pred_mask, gt_mask)
    l_seg = bce_seg + dice
    # L_bnd = BCE
    l_bnd = F.binary_cross_entropy_with_logits(pred_boundary, gt_boundary)
    return alpha * l_seg + beta * l_bnd, l_seg.detach(), l_bnd.detach()


@torch.no_grad()
def test(model, data_path, dataset, img_size):
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    test_loader = test_dataset(image_root, gt_root, img_size)
    num1 = test_loader.size

    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        out = model(image)  # (M,B) 或 M
        if isinstance(out, tuple) or isinstance(out, list):
            res = out[0]
        else:
            res = out

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        input_bin = (res >= 0.5)
        target = (np.array(gt) >= 0.5)
        smooth = 1
        input_flat = np.reshape(input_bin, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input_bin.sum() + target.sum() + smooth)
        DSC += float('{:.4f}'.format(dice))

    return DSC / num1, num1


def train(train_loader, model, optimizer, epoch, test_path, model_name='BGDNet', img_size=352, alpha=1.0, beta=0.4):
    model.train()
    global best
    global total_train_time
    global max_memory_usage
    max_memory_usage = 0
    time_before_epoch_start = time.time()
    size_rates = [1]
    loss_record = AvgMeter()
    loss_seg_meter = AvgMeter()
    loss_bnd_meter = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts, bnds = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            bnds = Variable(bnds).cuda()

            trainsize = int(round(img_size * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                bnds = F.interpolate(bnds, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            pred_m, pred_b = model(images)
            loss, l_seg, l_bnd = joint_loss(pred_m, gts, pred_b, bnds, alpha=alpha, beta=beta)

            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, args.batchsize)
                loss_seg_meter.update(l_seg.data, args.batchsize)
                loss_bnd_meter.update(l_bnd.data, args.batchsize)

            current_memory = torch.cuda.max_memory_allocated()
            max_memory_usage = max(max_memory_usage, current_memory)

        if i % 50 == 0 or i == total_step:
            inter_info = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], ' \
                         'loss: {:0.4f} (seg {:0.4f} | bnd {:0.4f}), max mem: {:.2f} MB'.format(
                datetime.now(), epoch, args.epoch, i, total_step,
                loss_record.show(), loss_seg_meter.show(), loss_bnd_meter.show(),
                max_memory_usage / (1024 ** 2)
            )
            print(inter_info)
            logging.info(inter_info)

    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: ' + str(total_train_time))
    logging.info('total train time till current epoch: ' + str(total_train_time))

    # save last
    save_path = (args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}-last.pth'))

    # evaluate
    if (epoch + 1) % 1 == 0:
        total_dice = 0
        total_images = 0
        for dataset in ['test']:
            dataset_dice, n_images = test(model, test_path, dataset, img_size=args.img_size)
            total_dice += (n_images * dataset_dice)
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)

        dataset_test_dice = total_dice / total_images
        print('Test dice score: {}'.format(dataset_test_dice))
        logging.info('Test dice score: {}'.format(dataset_test_dice))

        global best
        if dataset_test_dice > best:
            print(f'######## Dice improved {best:.4f} -> {dataset_test_dice:.4f}')
            logging.info(f'######## Dice improved {best:.4f} -> {dataset_test_dice:.4f}')
            best = dataset_test_dice
            torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}-best.pth'))

    torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=str, default='BGDNet', help='kept for compatibility')
    parser.add_argument("--num_classes", type=int, default=1, help="output channel of network")
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW or SGD')
    parser.add_argument('--augmentation', default=False, help='random flip rotation')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
    parser.add_argument('--img_size', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2/train/', help='train dataset root')
    parser.add_argument('--test_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2/test/', help='val dataset root')
    parser.add_argument('--train_save', type=str, default='./model_out/ISIC2018-best/BGDNet/')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for seg loss')
    parser.add_argument('--beta', type=float, default=0.4, help='weight for boundary loss')

    args = parser.parse_args()

    if not os.path.exists(args.train_save):
        os.makedirs(args.train_save)

    logging.basicConfig(filename=os.path.join(args.train_save, 'train.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build model ----
    model = BGDNet(num_classes=args.num_classes).cuda(0)
    logging.info(model)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    best = 0.0
    params = model.parameters()
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=1e-4, momentum=0.9)

    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)
    train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize, trainsize=args.img_size,
                              shuffle=True, augmentation=args.augmentation, drop_last=True)
    total_step = len(train_loader)
    globals()['total_step'] = total_step  # for logging
    print("#" * 20, "Start Training BGDNet", "#" * 20)
    total_train_time = 0

    for epoch in range(1, args.epoch):
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        train(train_loader, model, optimizer, epoch, args.test_path, model_name='BGDNet',
              img_size=args.img_size, alpha=args.alpha, beta=args.beta)

    print('avg train time: ' + str(total_train_time / (args.epoch - 1)))
    logging.info('avg train time: ' + str(total_train_time / (args.epoch - 1)))



