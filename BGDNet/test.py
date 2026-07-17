import torch
import torch.nn.functional as F
import os, argparse
import cv2
import numpy as np
import pandas as pd
from BGDNet.models.BGDNet_or import BGDNet
from utils.dataloader import test_dataset
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from matplotlib import pyplot as plt

def calculate_metrics(Y_test, yp):
    jacard = 0; dice = 0; tanimoto = 0; smooth = 1
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection
        only_neg = y2 * (1 - yp_2)
        only_pos = (1 - y2) * yp_2
        if (np.sum(y2) == 0) and (np.sum(yp_2) == 0):
            tanimoto += 1.0; jacard += 1.0; dice += 1.0
        elif (np.sum(intersection) == 0):
            tanimoto += 0.0; jacard += 0.0; dice += 0.0
        else:
            tanimoto += ((np.sum(intersection) + smooth) / (np.sum(intersection) + np.sum(only_neg) + np.sum(only_pos) + smooth))
            jacard += ((np.sum(intersection) + smooth) / (np.sum(union) + smooth))
            dice += (2. * np.sum(intersection) + smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)
    jacard /= len(Y_test); dice /= len(Y_test); tanimoto /= len(Y_test)
    return jacard, dice, tanimoto


def confusion_matrix_scorer(Y, Y_pred):
    Y = Y.astype(np.int8); Y_pred = Y_pred.astype(np.int8)
    P = len(np.where(Y == 1)[0]); N = len(np.where(Y == 0)[0])
    FP = len(np.where(Y - Y_pred == -1)[0])
    FN = len(np.where(Y - Y_pred == 1)[0])
    TP = len(np.where(Y + Y_pred == 2)[0])
    TN = len(np.where(Y + Y_pred == 0)[0])
    return P, N, TN, FP, FN, TP


def get_metrics(Y, pred):
    smooth = 1
    P = N = TN = FP = FN = TP = 0
    sensitivity = specificity = accuracy = precision = F1 = MCC = 0
    for i in range(len(Y)):
        _p, _n, _tn, _fp, _fn, _tp = confusion_matrix_scorer(Y[i], pred[i])
        P += _p; N += _n; TN += _tn; FP += _fp; FN += _fn; TP += _tp
        if (np.sum(Y[i]) == 0) and (np.sum(pred[i]) == 0):
            sensitivity += 1; specificity += 1; precision += 1; F1 += 1; MCC += 1
        else:
            if (_tp == 0):
                sensitivity += 0; precision += 0; F1 += 0.0
            else:
                sensitivity += (_tp / (_tp + _fn))
                precision += (_tp / (_tp + _fp))
                F1 += (2 * ((_tp / (_tp + _fp)) * (_tp / (_tp + _fn))) / ((_tp / (_tp + _fp)) + (_tp / (_tp + _fn))))
            if (_tn == 0): specificity += 0
            else: specificity += (_tn / (_tn + _fp))
            MCC += (_tp * _tn - _fp * _fn + smooth) / (np.power((_tp + _fp) * (_tp + _fn) * (_tn + _fp) * (_tn + _fn), 0.5) + smooth)
        accuracy += ((_tp + _tn) / (_tp + _fn + _fp + _tn))
    return P, N, TN, FP, FN, TP, sensitivity / len(Y), specificity / len(Y), accuracy / len(Y), precision / len(Y), F1 / len(Y), MCC / len(Y)


def get_metrics_and_print(Y, yp, method="BGDNet", testset='test', threshold=0.5, show=False, write=False):
    P, N, TN, FP, FN, TP, sensitivity, specificity, accuracy, precision, f1, mcc_cal = get_metrics(Y, yp)
    jacard, dice, tanimoto = calculate_metrics(Y, yp)
    cmat = [[TN, FP], [FN, TP]]
    cmat_score = [[TN / N, FP / N], [FN / P, TP / P]]
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall (Sensitivity):", sensitivity)
    print("F1 Score:", f1)
    print("MCC:", mcc_cal)
    print('Dice:', dice)
    print('Jacard:', jacard)
    print('Tanimoto:', tanimoto)
    if write:
        results = pd.DataFrame([[method, TN, FP, FN, TP, jacard, dice, sensitivity, specificity, accuracy, precision, f1, mcc_cal]],
                               columns=['Method', 'TN', 'FP', 'FN', 'TP', 'mIoU/Jacard', 'DICE', 'Sensitivity/Recall', 'Specificity', 'Accuracy', 'Precision', 'F-score', 'MCC'])
        results.to_csv('Repeat_results_' + testset + '.csv', mode='a', index=False, header=False)

if __name__ == '__main__':
    method_name = 'BGDNet'
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--pth_path', type=str, default='/zjy_work/MSDUNet-main/model_out/PH2-1/BGDNet/BGDNet-best.pth')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save_csv', type=str, default='./per_image_metrics.csv', help="Path to save per-image Dice/IoU")
    # 👉 新增：汇总指标 CSV 输出路径
    parser.add_argument('--save_summary_csv', type=str, default='./summary_metrics.csv',
                        help="Path to save summary metrics: Sensitivity, Specificity, Accuracy, Jacard, Dice")
    args = parser.parse_args()

    model = BGDNet(num_classes=args.num_classes).cuda(0)
    sd = torch.load(args.pth_path, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()

    dataset_name = 'ISIC2018'
    for _data_name in ['test']:
        data_path = '/zjy_work/PH2/{}'.format(_data_name)
        save_path = './model_out/PH2-1/{}/{}_{{}}/'.format(method_name, dataset_name)
        mask_dir = save_path.format('masks')
        bnd_dir = save_path.format('boundaries')
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(bnd_dir, exist_ok=True)

        print('Evaluating ' + data_path)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, args.testsize)

        DSC = 0.0
        JACARD = 0.0
        preds = []
        gts = []
        per_image_records = []

        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            with torch.no_grad():
                pm, pb = model(image)

            pm = F.interpolate(pm, size=gt.shape, mode='bilinear', align_corners=False)
            pm = torch.sigmoid(pm).data.cpu().numpy().squeeze()
            pm = (pm - pm.min()) / (pm.max() - pm.min() + 1e-8)
            cv2.imwrite(os.path.join(mask_dir, name), pm * 255)

            pb = F.interpolate(pb, size=gt.shape, mode='bilinear', align_corners=False)
            pb = torch.sigmoid(pb).data.cpu().numpy().squeeze()
            pb = (pb - pb.min()) / (pb.max() - pb.min() + 1e-8)
            cv2.imwrite(os.path.join(bnd_dir, name), pb * 255)

            input_bin = (pm >= 0.5).astype(np.uint8)
            target = (np.array(gt) >= 0.5).astype(np.uint8)

            preds.append(input_bin)
            # ✅ 修正：保存二值 GT（而非原始浮点掩码）
            gts.append(target)

            smooth = 1
            input_flat = np.reshape(input_bin, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            union = input_flat + target_flat - intersection

            jacard = ((np.sum(intersection) + smooth) / (np.sum(union) + smooth))
            dice = (2 * intersection.sum() + smooth) / (input_bin.sum() + target.sum() + smooth)

            JACARD += float('{:.4f}'.format(jacard))
            DSC += float('{:.4f}'.format(dice))

            per_image_records.append({
                "image_name": name,
                "dice": float('{:.4f}'.format(dice)),
                "iou": float('{:.4f}'.format(jacard))
            })

        print('*****************************************************')
        print('Dice Score: ' + str(DSC / num1))
        print('Jacard Score: ' + str(JACARD / num1))
        print(_data_name, 'Finish!')
        print('*****************************************************')

        # 逐图指标 CSV（保持原逻辑）
        df = pd.DataFrame(per_image_records)
        df.to_csv(args.save_csv, index=False)
        print(f"Per-image metrics saved to {args.save_csv}")

        # ========= 新增：汇总指标（Sensitivity、Specificity、Accuracy、Jacard、Dice）保存为 CSV =========
        # 使用你已实现的评估函数对整个数据集进行计算
        _, _, _, _, _, _, sensitivity, specificity, accuracy, _, _, _ = get_metrics(gts, preds)
        jacard_mean, dice_mean, _ = calculate_metrics(gts, preds)

        summary_df = pd.DataFrame([{
            "Sensitivity": float(f"{sensitivity:.6f}"),
            "Specificity": float(f"{specificity:.6f}"),
            "Accuracy":   float(f"{accuracy:.6f}"),
            "Jacard":     float(f"{jacard_mean:.6f}"),
            "Dice":       float(f"{dice_mean:.6f}")
        }])

        # 如果你希望每次评估都追加，可以把 mode='a' 且 header=not os.path.exists(...)
        summary_df.to_csv(args.save_summary_csv, index=False)
        print(f"Summary metrics saved to {args.save_summary_csv}")
