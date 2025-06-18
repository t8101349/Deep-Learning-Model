import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth= 1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0),-1)
    target_flat = target.view(target.size(0),-1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth)/ (pred_flat.sum(1) + target_flat.sum(1)+ smooth) 
    return 1 - dice.mean()

def iou_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0),-1)
    target_flat = target.view(target.size(0),-1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(1) + target_flat.sum(1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1-iou.mean()


def binary_focal_loss(pred, target, alpha=0.8, gamma=2):
    pred = torch.sigmoid(pred)
    target = target.type_as(pred)
    BCE= F.binary_cross_entropy(pred, target,reduction='none')
    pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
    # pt = torch.exp(-BCE)
    F_loss = alpha * (1-pt) ** gamma * BCE
    return F_loss.mean()

def combo_loss(pred, target):
    return 0.5*dice_loss(pred, target) + 0.5*binary_focal_loss(pred, target)

    
def iou_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    inter = (pred & target).float().sum((1, 2, 3))
    union = (pred | target).float().sum((1, 2, 3))
    return (inter / (union + 1e-6)).mean()