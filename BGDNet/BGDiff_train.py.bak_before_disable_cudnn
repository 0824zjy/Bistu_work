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

from models.BGDNet import BGDNet
from utils.dataloader_BGDiff import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

warnings.filterwarnings("ignore")


def dice_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + eps
    dice = 1.0 - (num / den)
    return dice.mean()

def joint_loss(pred_m, gts, pred_b, bnds, alpha=1.0, beta=1.0):
    pred_m = F.interpolate(pred_m, size=gts.shape[2:], mode='bilinear', align_corners=False)
    pred_b = F.interpolate(pred_b, size=bnds.shape[2:], mode='bilinear', align_corners=False)

    bce_seg = F.binary_cross_entropy_with_logits(pred_m, gts)
    bce_bnd = F.binary_cross_entropy_with_logits(pred_b, bnds)
    loss = alpha * bce_seg + beta * bce_bnd
    return loss, bce_seg, bce_bnd

@torch.no_grad()
def test(model, data_path, img_size, test_list=None):
    image_root = '{}/Images/'.format(data_path)
    gt_root = '{}/Masks/'.format(data_path)

    model.eval()
    test_loader = test_dataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=img_size,
        list_txt=test_list,
        mode="isic",
    )

    num1 = test_loader.size
    if num1 == 0:
        print("[WARN] test_loader.size == 0, 请检查 test_list 与 test root 路径/命名规则")
        return 0.0, 0

    # 全程用 torch(GPU) 累加
    DSC = torch.zeros((), device="cuda", dtype=torch.float32)
    smooth = 1.0

    for _ in range(num1):
        image, gt, name = test_loader.load_data()

        # gt: 仍然按你原先的方式读出来，但马上转 GPU tensor，并在 GPU 上归一化 + 二值化
        gt_np = np.asarray(gt, np.float32)
        gt_t = torch.from_numpy(gt_np).to(device="cuda", dtype=torch.float32)

        # 归一化（等价于你原来的 gt /= (gt.max() + 1e-8)）
        gt_max = gt_t.max()
        gt_t = gt_t / (gt_max + 1e-8)

        image = image.cuda()

        out = model(image)
        res = out[0] if isinstance(out, (tuple, list)) else out

        # res: GPU 上插值 + sigmoid
        res = F.interpolate(res, size=gt_t.shape, mode='bilinear', align_corners=False)
        res = torch.sigmoid(res)

        # res 可能是 [1,1,H,W]，压到 [H,W] 以便和 gt_t 对齐
        res = res.squeeze()

        # 等价于你原来的 min-max 归一化，但在 GPU 上做
        res_min = res.min()
        res_max = res.max()
        res = (res - res_min) / (res_max - res_min + 1e-8)

        # 二值化（GPU）
        input_bin = (res >= 0.5).to(torch.float32)
        target_bin = (gt_t >= 0.5).to(torch.float32)

        # Dice（GPU）
        intersection = (input_bin * target_bin).sum()
        dice = (2.0 * intersection + smooth) / (input_bin.sum() + target_bin.sum() + smooth)

        DSC += dice

    # 返回 float（只在最后把标量搬回 CPU）
    return (DSC / num1).item(), num1

def train(train_loader, model, optimizer, epoch, test_path, img_size=352, alpha=1.0, beta=0.4):
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
            # images = Variable(images).cuda()
            # gts = Variable(gts).cuda()
            # bnds = Variable(bnds).cuda()
            images = images.cuda(non_blocking=True)
            gts    = gts.cuda(non_blocking=True)
            bnds   = bnds.cuda(non_blocking=True)

            trainsize = int(round(img_size * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
                bnds = F.interpolate(bnds, size=(trainsize, trainsize), mode='bilinear', align_corners=False)

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
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f'BGDNet-last.pth'))

    # evaluate (每个epoch都测)
    dataset_dice, n_images = test(model, test_path, img_size=args.img_size, test_list=args.test_list)
    if n_images > 0:
        print('Test dice score: {}'.format(dataset_dice))
        logging.info('Test dice score: {}'.format(dataset_dice))

        global best
        if dataset_dice > best:
            print(f'######## Dice improved {best:.4f} -> {dataset_dice:.4f}')
            logging.info(f'######## Dice improved {best:.4f} -> {dataset_dice:.4f}')
            best = dataset_dice
            torch.save(model.state_dict(), os.path.join(save_path, f'BGDNet-best.pth'))

    torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=str, default='BGDNet', help='kept for compatibility')
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--augmentation', default=False)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=200)

    parser.add_argument('--train_path1', type=str, default='/data/zjy_work/ISIC2018/train/', help='真实训练集根路径')
    parser.add_argument('--train_path2', type=str, default='/data/zjy_work/ISIC2018/exp_mask2img_cn_only/', help='生成训练集根路径')
    parser.add_argument('--test_path', type=str, default='/data/zjy_work/ISIC2018/test/', help='测试集根路径')

    parser.add_argument('--train_list1', type=str, default=None)
    parser.add_argument('--train_list2', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)

    parser.add_argument('--train_save', type=str, default='./model_out/ISIC2018/BGDNet/')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.4)

    args = parser.parse_args()

    os.makedirs(args.train_save, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.train_save, 'train.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p'
    )

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

    # ====== 可选数据源：默认只用 train_path1/train_list1 ======
    image_roots = [os.path.join(args.train_path1, 'Images')]
    gt_roots    = [os.path.join(args.train_path1, 'Masks')]
    list_txts   = [args.train_list1]
    modes       = ["isic"]

    # 当且仅当你显式提供 train_list2 + train_path2 时，才启用第二套数据
    use_second = (
        args.train_list2 is not None and args.train_list2 != "" and
        args.train_path2 is not None and args.train_path2 != ""
    )

    if use_second:
        image_roots.append(os.path.join(args.train_path2, 'images'))
        gt_roots.append(os.path.join(args.train_path2, 'masks'))
        list_txts.append(args.train_list2)
        modes.append("paired")

    train_loader = get_loader(
        image_roots=image_roots,
        gt_roots=gt_roots,
        batchsize=args.batchsize,
        trainsize=args.img_size,
        shuffle=True,
        augmentation=args.augmentation,
        drop_last=True,
        list_txts=list_txts,
        modes=modes
    )

    print(f"[INFO] Using data sources: {len(image_roots)}")
    print(f"[INFO] image_roots = {image_roots}")
    print(f"[INFO] gt_roots    = {gt_roots}")
    print(f"[INFO] list_txts   = {list_txts}")
    print(f"[INFO] modes       = {modes}")
    # ==========================================================


    total_step = len(train_loader)
    globals()['total_step'] = total_step
    print("#" * 20, "Start Training BGDNet", "#" * 20)
    print(f"合并训练集路径: {image_roots}")
    print(f"训练集总批次: {total_step}")

    total_train_time = 0.0

    for epoch in range(1, args.epoch + 1):
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        train(train_loader, model, optimizer, epoch, args.test_path,
              img_size=args.img_size, alpha=args.alpha, beta=args.beta)

    print('avg train time: ' + str(total_train_time / args.epoch))
    logging.info('avg train time: ' + str(total_train_time / args.epoch))
