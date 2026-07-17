import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2

class DynamicAugment:
    def __init__(self, intensity=0.5):
        self.intensity = intensity
        
    def __call__(self, img, mask):
        # 随机选择增强方式
        aug_type = random.choice(['color', 'spatial', 'noise', 'none'])
        
        if aug_type == 'color' and random.random() < self.intensity:
            # 颜色增强
            img = self.color_jitter(img)
            
        if aug_type == 'spatial' and random.random() < self.intensity:
            # 空间增强
            img, mask = self.spatial_jitter(img, mask)
            
        if aug_type == 'noise' and random.random() < self.intensity:
            # 噪声增强
            img = self.add_noise(img)
            
        return img, mask
    
    def color_jitter(self, img):
        # 随机颜色扰动
        img = np.array(img)
        h, w, c = img.shape
        
        # HSV空间扰动
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        
        # 色调扰动
        hue_shift = random.uniform(-0.1, 0.1)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
        
        # 饱和度扰动
        sat_scale = random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        
        # 亮度扰动
        val_scale = random.uniform(0.8, 1.2)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
        
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(img)
    
    def spatial_jitter(self, img, mask):
        # 弹性变换
        img = np.array(img)
        mask = np.array(mask)
        
        # 检查图像尺寸是否过大
        if img.shape[0] > 10000 or img.shape[1] > 10000:
            # 如果图像过大，跳过弹性变换
            return Image.fromarray(img), Image.fromarray(mask)
        
        # 生成随机位移场
        alpha = random.uniform(100, 300)  # 减小位移幅度
        sigma = random.uniform(10, 15)
        random_state = np.random.RandomState(None)
        
        shape = img.shape[:2]
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        
        # 平滑位移场
        dx = cv2.GaussianBlur(dx, (17, 17), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (17, 17), sigma) * alpha
        
        # 生成网格
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
        
        try:
            # 应用位移
            img = cv2.remap(img, indices[1].astype(np.float32), indices[0].astype(np.float32), cv2.INTER_LINEAR)
            mask = cv2.remap(mask, indices[1].astype(np.float32), indices[0].astype(np.float32), cv2.INTER_NEAREST)
        except cv2.error:
            # 如果发生错误，返回原始图像
            return Image.fromarray(img), Image.fromarray(mask)
        
        return Image.fromarray(img), Image.fromarray(mask)
    
    def add_noise(self, img):
        # 添加混合噪声
        img = np.array(img)
        noise_type = random.choice(['gaussian', 'poisson', 'speckle'])
        
        if noise_type == 'gaussian':
            mean = 0
            var = random.uniform(0.001, 0.005)
            noise = np.random.normal(mean, var**0.5, img.shape) * 255
            img = img + noise
        elif noise_type == 'poisson':
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            img = np.random.poisson(img * vals) / float(vals)
        elif noise_type == 'speckle':
            noise = np.random.randn(*img.shape)
            img = img + img * noise * 0.1
            
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

class PolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.dynamic_augment = DynamicAugment(intensity=0.7)
        
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) 
                      if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) 
                   if f.endswith('.png') or f.endswith('.jpg')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        
        if self.augmentations:
            print('Using Dynamic Augmentation')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        # 动态增强
        if self.augmentations and random.random() > 0.5:
            image, gt = self.dynamic_augment(image, gt)
        
        seed = np.random.randint(1998)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)
        
        return image, gt
    
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=False, num_workers=2, pin_memory=True, augmentation=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) 
                      if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) 
                   if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')