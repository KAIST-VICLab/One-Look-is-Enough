# The code is written based on the code from https://gist.github.com/ranftlr/1d6194db2e1dffa0a50c9b0a9549cbd2
# Modified by: Byeongjun Kwon, 2025

import copy
import kornia

import torch
import torch.nn as nn
from mmengine import print_log
import torch.nn.functional as F
import random
import math

from estimator.registry import MODELS
from kornia.losses import dice_loss, focal_loss
from typing import Union, List

# Use the mask that combines the unreliable mask and the edge mask
@MODELS.register_module()
class AffineInvariantLoss_dict_unrel(nn.Module):
    def __init__(self, data_loss='mse_loss', scales=4, reduction="batch-based", align_corners=True, loss_weights={'mae_loss': 1.0, 'mse_loss':1.0, 'grad_loss': 5.0, 'weight_loss': 2.0}):
        super(AffineInvariantLoss_dict_unrel, self).__init__()
        
        # Support multiple data losses
        self.multiscale_laplace_loss = None
        self.data_loss_funcs = {}
        if isinstance(data_loss, str):
            data_loss = [data_loss]
        for loss in data_loss:
            if loss == 'mse_loss':
                self.data_loss_funcs['mse_loss'] = mse_loss
            elif loss == 'mae_loss':
                self.data_loss_funcs['mae_loss'] = mae_loss
            elif loss == 'trimmed_mae_loss':
                self.data_loss_funcs['trimmed_mae_loss'] = trimmed_mae_loss
        self.__scales = scales
        self.__align_corners = align_corners
        self.__loss_weights = loss_weights

    def forward(self, prediction, target, mask, median_gt, mad_gt, weight=None):
        _, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = target.shape
        
        # Align shapes if necessary
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=self.__align_corners)
            if weight is not None:
                weight = F.interpolate(weight, (h_t, w_t), mode='bilinear', align_corners=self.__align_corners)
                weight = weight.squeeze(1)
        
        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        mask = mask.squeeze(1)
        
        assert prediction.ndim == 3 and target.ndim == 3 and mask.ndim == 3, "prediction, target, mask must have 3 dimensions"
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding nan", logger='current')
            return {"total_loss": torch.sum(prediction * 0.0)}
           
        # Normalize target
        target_ = normalize_gt_robust(target, mask, median_gt, mad_gt)
        if target_ is None:
            return {"total_loss": torch.mean(prediction * 0.0)}
        
        # Compute data losses
        data_loss_dict = {}
        for loss_name, loss_func in self.data_loss_funcs.items():
            data_loss_dict[loss_name] = self.__loss_weights[loss_name] * loss_func(prediction, target_, mask)
        
        # Compute gradient matching loss
        grad_loss = 0.0
        if self.__loss_weights['grad_loss'] > 0:
            grad_loss = self.__loss_weights['grad_loss'] * multi_scale_gradient_matching_loss(
                prediction, target_, mask, scales=self.__scales
            )
        
        # Compute weighted loss if applicable
        weight_loss = 0.0
        if weight is not None:
            for loss_name, loss_func in self.data_loss_funcs.items():
                weight_loss += self.__loss_weights['weight_loss'] * loss_func(prediction, target_, mask, weight)
        
        # Return all losses as a dictionary
        losses = {
            **data_loss_dict,
            "grad_loss": grad_loss,
        }
        if weight is not None:
            losses["weight_loss"] = weight_loss
        losses["total_loss"] = sum(data_loss_dict.values()) + grad_loss + (weight_loss if weight is not None else 0.0)
        
        return losses

@MODELS.register_module()
class ConsistencyLoss(nn.Module):
    def __init__(self,
                 data_loss: Union[str,List[str]] = 'mse_loss',
                 loss_weights: dict = {'mae_loss': 1.0, 'mse_loss': 1.0},
                 ):
        super().__init__()
        self.data_loss_funcs = {}
        self.data_loss_dict = {}
        if isinstance(data_loss, str):
            data_loss = [data_loss]
        for loss in data_loss:
            if loss == 'mse_loss':
                self.data_loss_funcs['mse_loss'] = nn.MSELoss()
                self.data_loss_dict['mse_loss'] = 0.0
            elif loss == 'mae_loss':
                self.data_loss_funcs['mae_loss'] = nn.L1Loss()
                self.data_loss_dict['mae_loss'] = 0.0
        
        self.loss_weights = loss_weights
    
    def forward(self, prediction_list, rel_dict):
        for loss_name in self.data_loss_dict:
            self.data_loss_dict[loss_name] = 0.0
        
        for i in range(len(prediction_list)):
            for from_where, value in rel_dict[str(i)].items():
                for loss_name, loss_func in self.data_loss_funcs.items():
                    loss = loss_func(
                        prediction_list[i][:, :, value[0], value[1]],
                        prediction_list[int(from_where)][:,:, rel_dict[from_where][str(i)][0], rel_dict[from_where][str(i)][1]]
                    )
                    self.data_loss_dict[loss_name] += self.loss_weights[loss_name] * loss
        return sum(self.data_loss_dict.values())   
    

def gradient_matching_loss_mask(prediction, target, mask):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction_batch_based(image_loss, M)

def multi_scale_gradient_matching_loss(prediction, target, mask, scales=4):
    loss = 0.0
    for i in range(scales):
        step = 2 ** i
        loss += gradient_matching_loss_mask(prediction[:, ::step, ::step], target[:, ::step, ::step], mask[:, ::step, ::step])
       
    return loss


def mse_loss(prediction, target, mask, weight=None):
    N, H, W = prediction.shape
    M = torch.sum(mask, (1, 2))
    valid = M > 0
    res = (prediction[valid] - target[valid]) ** 2

    # res = res[mask.bool()].abs()
    res = (res * mask[valid]).abs()
    
    if weight is not None:
        res = res * weight[valid]

    return reduction_batch_based(res, M)

def mae_loss(prediction, target, mask, weight=None):
    N, H, W = prediction.shape
    M = torch.sum(mask, (1, 2))
    valid = M > 0
    res = prediction[valid] - target[valid]

    res = (res * mask[valid]).abs()
    
    if weight is not None:
        res = res * weight[valid]

    return reduction_batch_based(res, M)

def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    N, H, W = prediction.shape
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    # res = res[mask.bool()].abs()
    res = (res * mask).abs()

    trimmed = torch.sort(res.view(N,-1), descending=False, dim = -1)[0][
        :, 
        : int(H * W * (1.0 - trim))
    ]

    return reduction_batch_based(trimmed, M)

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def normalize_gt_robust(target, mask, median_gt, mad_gt):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if valid.sum() == 0:
        print("No valid pixels in batch")
        return None

    m = torch.zeros_like(ssum, dtype=torch.float32)
    s = torch.ones_like(ssum, dtype=torch.float32)
    
    m[valid] = median_gt[valid]
    target = target - m.view(-1, 1, 1)
    
    s[valid] = torch.clamp(mad_gt[valid], min=1e-6)

    return target / (s.view(-1, 1, 1))
