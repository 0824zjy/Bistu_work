import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred, target, features=None):
        # 结构损失
        structure_loss = self.structure_loss(pred, target)
        
        # 边界感知损失
        boundary_loss = self.boundary_aware_loss(pred, target)
        
        # 特征一致性损失
        consistency_loss = 0
        if features is not None:
            consistency_loss = self.feature_consistency(features)
        
        return self.alpha * structure_loss + self.beta * boundary_loss + self.gamma * consistency_loss
    
    def structure_loss(self, pred, target):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()
    
    def boundary_aware_loss(self, pred, target):
        # 计算边界图
        kernel = torch.ones(3, 3, device=target.device)
        boundary = F.conv2d(target, kernel.unsqueeze(0).unsqueeze(0), padding=1)
        boundary = torch.abs(boundary - 9 * target)
        
        # 边界加权损失
        pred_sig = torch.sigmoid(pred)
        diff = torch.abs(pred_sig - target)
        loss = boundary * diff
        return loss.mean()
    
    def feature_consistency(self, features):
        # 特征一致性约束
        loss = 0
        for i in range(1, len(features)):
            feat1 = F.normalize(features[i-1], dim=1)
            feat2 = F.normalize(features[i], dim=1)
            loss += 1 - F.cosine_similarity(feat1, feat2, dim=1).mean()
        return loss / (len(features) - 1)