import os
import random
import shutil

if __name__ == '__main__':
    root = '/zjy_work/ISIC2018'
    save = '{}/'.format(root)

    image_root = '{}/ISIC2018_Task1-2_Training_Input/'.format(root)
    gt_root = '{}/ISIC2018_Task1_Training_GroundTruth/'.format(root)

    save_trainpath = os.path.join(save, 'train')
    save_valpath = os.path.join(save, 'val')
    save_testpath = os.path.join(save, 'test')
    save_path = [save_trainpath, save_valpath, save_testpath]

    images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
    images = sorted(images)
    gts = sorted(gts)

    index = list(range(len(images)))
    random.shuffle(index)

    train_index = index[:int(0.8 * len(images))]
    val_index = index[int(0.8 * len(images)):int(0.9 * len(images))]
    test_index = index[int(0.9 * len(images)):]

    source, mask = [], []
    for i, index in enumerate([train_index, val_index, test_index]):
        for idx in index:
            img = images[idx]
            imageName = os.path.basename(img).split('.')[0]
            gt_img = gts[idx]
            gt_imageName = os.path.basename(gt_img).split('.')[0].split('_seg')[0]

            if imageName != gt_imageName:
                continue

            dst_image_path = os.path.join(save_path[i], 'images')
            dst_mask_path = os.path.join(save_path[i], 'masks')

            os.makedirs(dst_image_path, exist_ok=True)
            os.makedirs(dst_mask_path, exist_ok=True)

            dst_image_name = os.path.join(dst_image_path, imageName + '.jpg')
            dst_mask_name = os.path.join(dst_mask_path, imageName + '.jpg')

            shutil.copy(img, dst_image_name)
            shutil.copy(gt_img, dst_mask_name)
        print(save_path[i])
