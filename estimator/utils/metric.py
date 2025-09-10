import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import canny
import kornia
import copy
from d3r.d3r import ord
from typing import Tuple, List

# code from https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d
def compute_scale_and_shift(prediction, target, mask):
    # prediction shape: (B, H, W), target shape: (B, H, W), mask shape: (B, H, W)
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def compute_errors_torch(prediction, target, mask, depth_cap=80.0, threshold=1.25):
    # prediction shape: (B, H, W), target shape: (B, H, W), mask shape: (B, H, W)
    # target: depth_gt, prediction: disp_pred
    # mask: valid_mask
    
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
    # scale, shift = scale.float(), shift.float()
    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    disparity_cap = 1.0 / depth_cap
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

    prediciton_depth = 1.0 / prediction_aligned

    # bad pixel
    err = torch.zeros_like(prediciton_depth, dtype=torch.float)

    err[mask == 1] = torch.max(
        prediciton_depth[mask == 1] / target[mask == 1],
        target[mask == 1] / prediciton_depth[mask == 1],
    ).float()

    err[mask == 1] = (err[mask == 1] > threshold).float()

    p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))
    
    abs_rel = torch.sum(torch.abs(prediciton_depth[mask == 1] - target[mask == 1]) / target[mask == 1]) / torch.sum(mask,(1, 2))

    return 100 * torch.mean(p), torch.mean(abs_rel)

def compute_depth_from_disp(prediction, target, mask, depth_cap=80.0):
    # prediction shape: (B, H, W), target shape: (B, H, W), mask shape: (B, H, W)
    # target: depth_gt, prediction: disp_pred
    # mask: valid_mask
    assert len(prediction.shape) == 3 and len(target.shape) == 3 and len(mask.shape) == 3
    
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
    # scale, shift = scale.float(), shift.float()
    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    disparity_cap = 1.0 / depth_cap
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

    prediciton_depth = 1.0 / prediction_aligned

    return prediciton_depth

def compute_depth_from_disp_minmax(prediction, target, mask, depth_max=80.0, depth_min=0.001):
    # prediction shape: (B, H, W), target shape: (B, H, W), mask shape: (B, H, W)
    # target: depth_gt, prediction: disp_pred
    # mask: valid_mask
    assert len(prediction.shape) == 3 and len(target.shape) == 3 and len(mask.shape) == 3
    
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
    # scale, shift = scale.float(), shift.float()
    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
    
    disp_min = 1.0 / depth_max
    disp_max = 1.0 / depth_min
    
    prediction_aligned[prediction_aligned < disp_min] = disp_min
    prediction_aligned[prediction_aligned > disp_max] = disp_max

    prediciton_depth = 1.0 / prediction_aligned

    return prediciton_depth

def compute_aligned_disp(prediction, target, depth_cap=80.0):
    # prediction shape: (B, H, W), target shape: (B, H, W), mask shape: (B, H, W)
    # target: disp_gt, prediction: disp_pred
    # return: aligned_disp_pred

    assert len(prediction.shape) == 3 and len(target.shape) == 3

    scale, shift = compute_scale_and_shift(prediction, target, torch.ones_like(target))

    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    disparity_cap = 1.0 / depth_cap
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

    return prediction_aligned
    

def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

def soft_edge_error(pred, gt, radius=1):
    abs_diff=[]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            abs_diff.append(np.abs(shift_2d_replace(gt, i, j, 0) - pred))
    return np.minimum.reduce(abs_diff)

def get_boundaries(disp, th=1., dilation=10):
    edges_y = np.logical_or(np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
                            np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))))
    edges_x = np.logical_or(np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
                            np.pad(np.abs(disp[:, :-1] - disp[:,1:]) > th, ((0, 0), (0, 1))))
    edges = np.logical_or(edges_y,  edges_x).astype(np.float32)

    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, disp_gt_edges=None, additional_mask=None):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            # pred, gt.shape[-2:], mode='bilinear', align_corners=True).squeeze()
            pred, gt.shape[-2:], mode='bilinear', align_corners=False).squeeze()

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    eval_mask = np.ones(valid_mask.shape)
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
            
    valid_mask = np.logical_and(valid_mask, eval_mask)
    
    # for prompt depth
    if additional_mask is not None:
        additional_mask = additional_mask.squeeze().detach().cpu().numpy()
        valid_mask = np.logical_and(valid_mask, additional_mask)
        
    metrics = compute_errors(gt_depth[valid_mask], pred[valid_mask])
    
    if disp_gt_edges is not None:
        
        edges = disp_gt_edges.squeeze().numpy()
        mask = valid_mask.squeeze() # squeeze
        mask = np.logical_and(mask, edges)

        see_depth = torch.tensor([0])
        if mask.sum() > 0:
            see_depth_map = soft_edge_error(pred, gt_depth)
            see_depth_map_valid = see_depth_map[mask]
            see_depth = see_depth_map_valid.mean()
        metrics['see'] = see_depth
    
    return metrics, (None, None)

def compute_metrics_d3r(gt, pred, valid_pairs, ord_pairs, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, disp_gt_edges=None, additional_mask=None):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            # pred, gt.shape[-2:], mode='bilinear', align_corners=True).squeeze()
            pred, gt.shape[-2:], mode='bilinear', align_corners=False).squeeze()

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    eval_mask = np.ones(valid_mask.shape)
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
            
    valid_mask = np.logical_and(valid_mask, eval_mask)
    
    # for prompt depth
    if additional_mask is not None:
        additional_mask = additional_mask.squeeze().detach().cpu().numpy()
        valid_mask = np.logical_and(valid_mask, additional_mask)
        
    metrics = compute_errors(gt_depth[valid_mask], pred[valid_mask])
    
    abs_rel_vis = np.zeros_like(gt_depth)
    abs_rel_vis[valid_mask] = np.abs(gt_depth[valid_mask] - pred[valid_mask]) / gt_depth[valid_mask]
    abs_rel_value = np.mean(abs_rel_vis[valid_mask])
    # abs_rel_vis[valid_mask] = np.clip(abs_rel_vis[valid_mask], 0, 1)
    # abs_rel_vis[valid_mask] = abs_rel_vis[valid_mask] * 255
    abs_rel_vis = (abs_rel_vis - np.min(abs_rel_vis)) / (np.max(abs_rel_vis) - np.min(abs_rel_vis))
    abs_rel_vis[~valid_mask] = 0
        
    if disp_gt_edges is not None:
        
        edges = disp_gt_edges.squeeze().numpy()
        mask = valid_mask.squeeze() # squeeze
        mask = np.logical_and(mask, edges)

        see_depth = torch.tensor([0])
        if mask.sum() > 0:
            see_depth_map = soft_edge_error(pred, gt_depth)
            see_depth_map_valid = see_depth_map[mask]
            see_depth = see_depth_map_valid.mean()
        metrics['see'] = see_depth
    
    error = 0
    confidence = 0
    valid_pairs = valid_pairs.squeeze().numpy()
    for valid_pair in valid_pairs:
        h1, w1, h2, w2 = valid_pair
        if not np.sum(valid_pair) == 0:
            error += np.abs(ord(pred[h1, w1], pred[h2, w2], 1.01) - ord(gt_depth[h1, w1], gt_depth[h2, w2], 1.01))
            confidence += 1
    error /= confidence
    metrics['d3r'] = error
    
    error = 0
    confidence = 0
    ord_pairs = ord_pairs.squeeze().numpy()
    for ord_pair in ord_pairs:
        h1, w1, h2, w2 = ord_pair
        if not np.sum(ord_pair) == 0:
            error += np.abs(ord(pred[h1, w1], pred[h2, w2], 1.03) - ord(gt_depth[h1, w1], gt_depth[h2, w2], 1.03))
            confidence += 1
    error /= confidence
    metrics['ord'] = error
    
    return metrics, (abs_rel_vis, abs_rel_value)


def eps(x):
    """Return the `eps` value for the given `input` dtype. (default=float32 ~= 1.19e-7)"""
    dtype = torch.float32 if x is None else x.dtype
    return torch.finfo(dtype).eps

def to_log(depth):
    """Convert linear depth into log depth."""
    depth = torch.tensor(depth)
    depth = (depth > 0) * depth.clamp(min=eps(depth)).log()
    return depth

def to_inv(depth):
    """Convert linear depth into disparity."""
    depth = torch.tensor(depth)
    disp = (depth > 0) / depth.clamp(min=eps(depth))
    return disp

def extract_edges(depth,
                  preprocess=None,
                  sigma=1,
                  mask=None,
                  use_canny=True):
    """Detect edges in a dense LiDAR depth map.

    :param depth: (ndarray) (h, w, 1) Dense depth map to extract edges.
    :param preprocess: (str) Additional depth map post-processing. (log, inv, none)
    :param sigma: (int) Gaussian blurring sigma.
    :param mask: (Optional[ndarray]) Optional boolean mask of valid pixels to keep.
    :param use_canny: (bool) If `True`, use `Canny` edge detection, otherwise `Sobel`.
    :return: (ndarray) (h, w) Detected depth edges in the image.
    """
    if preprocess not in {'log', 'inv', 'none', None}:
        raise ValueError(f'Invalid depth preprocessing. ({preprocess})')

    depth = depth.squeeze()
    if preprocess == 'log':
        depth = to_log(depth)
    elif preprocess == 'inv':
        depth = to_inv(depth)
        depth -= depth.min()
        depth /= depth.max()
    else:
        depth = torch.tensor(depth)
        input_value = (depth > 0) * depth.clamp(min=eps(depth))
        # depth = torch.log(input_value) / torch.log(torch.tensor(1.9))
        # depth = torch.log(input_value) / torch.log(torch.tensor(1.9))
        depth = torch.log(input_value) / torch.log(torch.tensor(1.5))
        
    depth = depth.numpy()

    if use_canny:
        edges = canny(depth, sigma=sigma, mask=mask)
    else:
        raise NotImplementedError("Sobel edge detection is not implemented yet.")

    return edges

# PatchRefiner code로부터 가져옴
def compute_boundary_metrics(
    gt, 
    pred, 
    gt_edges,
    valid_mask,
    pred_edges,
    metric_dict=None,
    th_edges_acc=10,
    th_edges_comp=10,):
    
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    valid_mask = valid_mask.numpy()
    invalid_mask = np.logical_not(valid_mask)
    gt_edges = gt_edges.numpy()
    pred_edges = pred_edges.numpy()
    gt_edges_copy = copy.deepcopy(gt_edges)
    pred_edges_copy = copy.deepcopy(pred_edges)
    
    # D_target = ndimage.distance_transform_edt(1 - gt_edge_update)
    D_target = ndimage.distance_transform_edt(np.logical_not(gt_edges))
    
    # D_pred = ndimage.distance_transform_edt(1 - pred_edges)  # Distance of each pixel to predicted edges
    D_pred = ndimage.distance_transform_edt(np.logical_not(pred_edges))  # Distance of each pixel to predicted edges
    
    gt_edges[invalid_mask] = 0
    pred_edges[invalid_mask] = 0
    
    pred_edges_BDE = pred_edges & (D_target < th_edges_acc)  # Predicted edges close enough to real ones. (This is from the offical repo)
    gt_edge_BDE = gt_edges & (D_pred < th_edges_comp)  # Real edges close enough to predicted ones.
    
    metric = {
        'EdgeAcc': D_target[pred_edges_BDE].mean() if pred_edges_BDE.sum() else th_edges_acc,  # Distance from pred to target
        # 'EdgeComp': D_pred[gt_edge_BDE].mean() if gt_edge_BDE.sum() else th_edges_comp,  # Distance from target to pred
        'EdgeComp': D_pred[gt_edges].mean() if pred_edges_BDE.sum() else th_edges_comp,  # Distance from target to pred
    }
    
    metric['EdgeAcc'] = torch.tensor(metric['EdgeAcc'])
    metric['EdgeComp'] = torch.tensor(metric['EdgeComp'])
    
    # loose the target (extend the edge area) when calculating the F1-score
    gt_edge_extend = \
        kornia.filters.gaussian_blur2d(torch.tensor(gt_edges_copy).unsqueeze(dim=0).unsqueeze(dim=0).float(), kernel_size=(5, 5), sigma=(5., 5.), border_type='reflect', separable=True)
    gt_edge_extend = gt_edge_extend > 0
    gt_edge_extend = gt_edge_extend.squeeze()
    gt_edge_extend = gt_edge_extend[valid_mask]
    
    pred_edge_extend = \
        kornia.filters.gaussian_blur2d(torch.tensor(pred_edges_copy).unsqueeze(dim=0).unsqueeze(dim=0).float(), kernel_size=(5, 5), sigma=(5., 5.), border_type='reflect', separable=True)
    pred_edge_extend = pred_edge_extend > 0
    pred_edge_extend = pred_edge_extend.squeeze()
    pred_edge_extend = pred_edge_extend[valid_mask]
    
    pred_edges = torch.tensor(pred_edge_extend)
    pred_edges_flat = pred_edges.view(-1).int().detach().cpu()
    gt_edge_flat = gt_edge_extend.view(-1).int().detach().cpu()
    
    for k, v in metric_dict.items():
        metric_value = v(pred_edges_flat, gt_edge_flat)
        metric[k] = metric_value
    
    return metric

# I get the following functions from Depth Pro (https://github.com/apple/ml-depth-pro) eval code.
# Modified by: Byeongjun Kwon, 2025
def get_thresholds_and_weights(
    t_min: float, t_max: float, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate thresholds and weights for the given range.

    Args:
    ----
        t_min (float): Minimum threshold.
        t_max (float): Maximum threshold.
        N (int): Number of thresholds.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: Array of thresholds and corresponding weights.

    """
    thresholds = np.linspace(t_min, t_max, N)
    weights = thresholds / thresholds.sum()
    return thresholds, weights

def fgbg_depth(
    d: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for comparison.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations.

    """
    right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
    left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
    bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
    top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )

def boundary_f1(
    pr: np.ndarray,
    gt: np.ndarray,
    t: float,
    return_p: bool = False,
    return_r: bool = False,
) -> float:
    """Calculate Boundary F1 score.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth depth matrix.
        t (float): Threshold for comparison.
        return_p (bool, optional): If True, return precision. Defaults to False.
        return_r (bool, optional): If True, return recall. Defaults to False.

    Returns:
    -------
        float: Boundary F1 score, or precision, or recall depending on the flags.

    """
    ap, bp, cp, dp = fgbg_depth(pr, t)
    ag, bg, cg, dg = fgbg_depth(gt, t)

    r = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )
    p = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ap), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bp), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cp), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dp), 1)
    )
    if r + p == 0:
        return 0.0
    if return_p:
        return p
    if return_r:
        return r
    return 2 * (r * p) / (r + p)


def SI_boundary_F1(
    predicted_disp: np.ndarray,
    target_disp: np.ndarray,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
) -> float:
    """Calculate Scale-Invariant Boundary F1 Score for disp-based ground-truth.
    In original code, the function compares predicted_depth and target_depth.
    Here, we use disp (disparity) instead.
    Modified by Byeongjun Kwon, 2025
    Args:
    ----
        predicted_disp (np.ndarray): Predicted disp matrix.
        target_disp (np.ndarray): Ground truth disp matrix.
        t_min (float, optional): Minimum threshold. Defaults to 1.05.
        t_max (float, optional): Maximum threshold. Defaults to 1.25.
        N (int, optional): Number of thresholds. Defaults to 10.

    Returns:
    -------
        float: Scale-Invariant Boundary F1 Score.

    """
    assert predicted_disp.ndim == target_disp.ndim == 2
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    f1_scores = np.array(
        [
            boundary_f1(predicted_disp, target_disp, t)
            for t in thresholds
        ]
    )
    return np.sum(f1_scores * weights)

def connected_component(r: np.ndarray, c: np.ndarray) -> List[List[int]]:
    """Find connected components in the given row and column indices.

    Args:
    ----
        r (np.ndarray): Row indices.
        c (np.ndarray): Column indices.

    Yields:
    ------
        List[int]: Indices of connected components.

    """
    indices = [0]
    for i in range(1, r.size):
        if r[i] == r[indices[-1]] and c[i] == c[indices[-1]] + 1:
            indices.append(i)
        else:
            yield indices
            indices = [i]
    yield indices

def nms_horizontal(ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Apply Non-Maximum Suppression (NMS) horizontally on the given ratio matrix.

    Args:
    ----
        ratio (np.ndarray): Input ratio matrix.
        threshold (float): Threshold for NMS.

    Returns:
    -------
        np.ndarray: Binary mask after applying NMS.

    """
    mask = np.zeros_like(ratio, dtype=bool)
    r, c = np.nonzero(ratio > threshold)
    if len(r) == 0:
        return mask
    for ids in connected_component(r, c):
        values = [ratio[r[i], c[i]] for i in ids]
        mi = np.argmax(values)
        mask[r[ids[mi]], c[ids[mi]]] = True
    return mask


def nms_vertical(ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Apply Non-Maximum Suppression (NMS) vertically on the given ratio matrix.

    Args:
    ----
        ratio (np.ndarray): Input ratio matrix.
        threshold (float): Threshold for NMS.

    Returns:
    -------
        np.ndarray: Binary mask after applying NMS.

    """
    return np.transpose(nms_horizontal(np.transpose(ratio), threshold))

def fgbg_depth_thinned(
    d: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels with Non-Maximum Suppression.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for NMS.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations with NMS applied.

    """
    #! 0인 값을 finfo(eps)로 바꿔주는 이유는 나중에 나누기 연산을 할 때 0으로 나누는 것을 방지하기 위함
    d = np.maximum(d, np.finfo(d.dtype).eps)
    right_is_big_enough = nms_horizontal(d[..., :, 1:] / d[..., :, :-1], t)
    left_is_big_enough = nms_horizontal(d[..., :, :-1] / d[..., :, 1:], t)
    bottom_is_big_enough = nms_vertical(d[..., 1:, :] / d[..., :-1, :], t)
    top_is_big_enough = nms_vertical(d[..., :-1, :] / d[..., 1:, :], t)
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )

def fgbg_binary_mask(
    d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels in binary masks.

    Args:
    ----
        d (np.ndarray): Binary depth matrix.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations in binary masks.

    """
    assert d.dtype == bool
    right_is_big_enough = d[..., :, 1:] & ~d[..., :, :-1]
    left_is_big_enough = d[..., :, :-1] & ~d[..., :, 1:]
    bottom_is_big_enough = d[..., 1:, :] & ~d[..., :-1, :]
    top_is_big_enough = d[..., :-1, :] & ~d[..., 1:, :]
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )

def edge_recall_matting(pr: np.ndarray, gt: np.ndarray, t: float) -> float:
    """Calculate edge recall for image matting.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth binary mask.
        t (float): Threshold for NMS.

    Returns:
    -------
        float: Edge recall value.

    """
    assert gt.dtype == bool
    ap, bp, cp, dp = fgbg_depth_thinned(pr, t)
    ag, bg, cg, dg = fgbg_binary_mask(gt)
    return 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )

def SI_boundary_Recall(
    predicted_disp: np.ndarray,
    target_mask: np.ndarray,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
    alpha_threshold: float = 0.1,
) -> float:
    """Calculate Scale-Invariant Boundary Recall Score for mask-based ground-truth.

    Args:
    ----
        predicted_disp (np.ndarray): Predicted disparity matrix.
        target_mask (np.ndarray): Ground truth binary mask.
        t_min (float, optional): Minimum threshold. Defaults to 1.05.
        t_max (float, optional): Maximum threshold. Defaults to 1.25.
        N (int, optional): Number of thresholds. Defaults to 10.
        alpha_threshold (float, optional): Threshold for alpha masking. Defaults to 0.1.

    Returns:
    -------
        float: Scale-Invariant Boundary Recall Score.

    """
    assert predicted_disp.ndim == target_mask.ndim == 2
    assert predicted_disp.shape == target_mask.shape
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    thresholded_target = target_mask > alpha_threshold

    recall_scores = np.array(
        [
            edge_recall_matting(
                predicted_disp, thresholded_target, t=float(t)
            )
            for t in thresholds
        ]
    )
    weighted_recall = np.sum(recall_scores * weights)
    return weighted_recall