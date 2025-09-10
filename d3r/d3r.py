# depth discontinuity disagreement ratio

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import slic, mark_boundaries


def ord(val1, val2, delta):
    ratio = (val1 + np.finfo(float).eps) / (val2 + np.finfo(float).eps)
    if ratio > delta:
        return 1
    elif ratio < 1 / delta:
        return -1
    else:
        return 0

def extract_d3r_points(gt, samples, compactness=0.001):
    gt[np.isnan(gt)] = 0
    segments = slic(gt, n_segments=samples, compactness=compactness, start_label=0, channel_axis=None)
    num_labels = segments.max()+1

    height, width = gt.shape
    neighbouring_rel = [[] for _ in range(num_labels)]
    random_rel = [[] for _ in range(num_labels)]
    # centers = np.zeros(num_labels, dtype=int)
    centers = [None] * num_labels

    for i in range(num_labels):
        mask = segments == i
        distance = distance_transform_edt(~mask)
        # segments에서 neighbor pixel의 위치를 저장한게 아니라 값을 저장한거 아닌가?
        neighbours = np.unique(segments[distance == 1])
        neighbouring_rel[i] = neighbours.tolist()
        random_rel[i] = np.random.choice(range(num_labels), size=3, replace=True)

        # Find the center of the superpixel
        pixel_indices = np.argwhere(mask)
        center = np.mean(pixel_indices, axis=0).astype(int)
        # centers[i] = np.ravel_multi_index(center, (height, width))
        centers[i] = tuple(center)

    return centers, neighbouring_rel, random_rel, segments


def d3r(gt, depth_est, center_points, point_pairs, freq_ratio, d3r_ratio):
    """
    Python equivalent of the MATLAB D3R function.
    
    Args:
        gt: Ground truth array.
        depth_est: Depth estimation array.
        center_points: Superpixel center locations.
        point_pairs: Point pair relations.
        freq_ratio: Frequency ratio.
        d3r_ratio: Metric ratio.

    Returns:
        confidence: Number of point pairs used.
        error: Sum of errors.
    """
    gt = np.nan_to_num(gt)  # Replace NaNs with 0
    inflamed_gt = gt  # Adjustments, if any, can be applied here

    same_ratio_gt = 1 + d3r_ratio
    same_ratio_est = 1 + d3r_ratio

    error = 0
    confidence = 0

    for i in range(len(center_points)):
        neighbours = point_pairs[i]
        if not neighbours:
            continue
        
        for j in neighbours:
            j_neighbours = point_pairs[j]
            if i in j_neighbours:
                j_neighbours.remove(i)

            point_pairs[j] = j_neighbours if j_neighbours else []

            index1 = center_points[i]
            index2 = center_points[j]

            if gt[index1] != 0 and gt[index2] != 0:  # Check for error in GT
                if ord(gt[index1], gt[index2], 1 + freq_ratio) != 0:
                    er = abs(
                        ord(inflamed_gt[index1], inflamed_gt[index2], same_ratio_gt) - 
                        ord(depth_est[index1], depth_est[index2], same_ratio_est)
                    )
                    error += er
                    confidence += 1

    return confidence, error

def d3r_validpair(gt, center_points, point_pairs, freq_ratio, d3r_ratio):
    """
    Python equivalent of the MATLAB D3R function.
    
    Args:
        gt: Ground truth array.
        depth_est: Depth estimation array.
        center_points: Superpixel center locations.
        point_pairs: Point pair relations.
        freq_ratio: Frequency ratio.
        d3r_ratio: Metric ratio.

    Returns:
        confidence: Number of point pairs used.
        error: Sum of errors.
    """
    gt = np.nan_to_num(gt)  # Replace NaNs with 0
    inflamed_gt = gt  # Adjustments, if any, can be applied here

    same_ratio_gt = 1 + d3r_ratio
    same_ratio_est = 1 + d3r_ratio

    # error = 0
    # confidence = 0
    
    valid_pairs = []

    for i in range(len(center_points)):
        neighbours = point_pairs[i]
        if not neighbours:
            continue
        
        for j in neighbours:
            j_neighbours = point_pairs[j]
            if i in j_neighbours:
                j_neighbours.remove(i)

            point_pairs[j] = j_neighbours if j_neighbours else []

            index1 = center_points[i]
            index2 = center_points[j]

            if gt[index1] != 0 and gt[index2] != 0:  # Check for error in GT
                if ord(gt[index1], gt[index2], 1 + freq_ratio) != 0:
                    valid_pairs.append((i, j))
                    # er = abs(
                    #     ord(inflamed_gt[index1], inflamed_gt[index2], same_ratio_gt) - 
                    #     ord(depth_est[index1], depth_est[index2], same_ratio_est)
                    # )
                    # error += er
                    # confidence += 1

    return valid_pairs

import cv2
import os
from os import listdir
from os.path import join
import numpy as np

def evaluate_d3r(gt_depth_path, estimation_path, superpixel_scale=1, samples=5000, type='disp'):
    """
    평가 메트릭 (D3R 포함)을 계산하는 함수.

    Parameters:
        gt_depth_path (str): Ground truth 깊이 맵 경로.
        estimation_path (str): 예측 깊이 맵 경로.
        superpixel_scale (float): superpixel 비율 조정.
        samples (int): SLIC 초매개변수.

    Returns:
        dict: 평가 메트릭 결과.
    """
    img_list = [f for f in listdir(gt_depth_path) if f.endswith('.png')]

    # 메트릭 리스트 초기화
    d3r_errors = []

    for img_name in img_list:
        if type == 'disp':
            gt_disp = cv2.imread(join(gt_depth_path, img_name), cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_disp = (gt_disp - np.nanmin(gt_disp)) / (np.nanmax(gt_disp) - np.nanmin(gt_disp))

        est_img = cv2.imread(join(estimation_path, img_name), cv2.IMREAD_UNCHANGED).astype(np.float64)
        est_disp = (est_img - np.nanmin(est_img)) / (np.nanmax(est_img) - np.nanmin(est_img))

        est_depth = 1 / est_disp
        est_depth = (est_depth - np.nanmin(est_depth)) / (np.nanmax(est_depth) - np.nanmin(est_depth))

        # SLIC에서 D3R 포인트 추출
        gt_small = cv2.resize(gt_disp, None, fx=superpixel_scale, fy=superpixel_scale, interpolation=cv2.INTER_NEAREST)
        centers, neighbouring_rel, random_rel, segments = extract_d3r_points(gt_small, samples)

        # D3R 계산
        confidence, error = d3r(gt_disp, est_depth, centers, neighbouring_rel, 0.1, 0.01)
        d3r_errors.append(error / confidence if confidence else np.nan)


    # 결과 반환
    return {
        'd3r_errors': d3r_errors,
    }

def visualize_d3r(gt_path, superpixel_scale=1, samples=5000, type='disp', compactness=0.001):
    """
    D3R 조건을 만족하는 페어와 superpixel 경계 시각화.
    
    Args:
        gt: Ground truth disparity map (normalized 2D array).
        segments: Superpixel segment labels.
        center_points: Superpixel 중심점 좌표.
        point_pairs: Superpixel 간의 관계 (이웃 페어).
        valid_pairs: D3R 조건을 만족하는 페어 리스트 [(index1, index2), ...].
    """
    if type == 'disp':
        gt_disp = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        gt_disp = (gt_disp - np.nanmin(gt_disp)) / (np.nanmax(gt_disp) - np.nanmin(gt_disp))
        gt = gt_disp
    
    gt_small = cv2.resize(gt, None, fx=superpixel_scale, fy=superpixel_scale, interpolation=cv2.INTER_NEAREST)
    centers, neighbouring_rel, random_rel, segments = extract_d3r_points(gt_small, samples, compactness)
    
    centers = [(int(center[0] / superpixel_scale), int(center[1] / superpixel_scale)) for center in centers]
    
    freq_ratio = 0.1
    d3r_ratio = 0.01
    
    valid_pairs = d3r_validpair(gt, centers, neighbouring_rel, freq_ratio, d3r_ratio)
    
    # 원본 이미지 복사
    visualization = (gt * 255).astype(np.uint8)  # 흑백으로 변환
    visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)  # RGB로 변환
    
    segments = cv2.resize(segments, None, fx=1/superpixel_scale, fy=1/superpixel_scale, interpolation=cv2.INTER_NEAREST)
    # 경계선 표시 (파란색)
    boundaries = mark_boundaries(gt, segments, color=(1, 1, 0), mode='outer', outline_color=(1,1,0))
    boundaries = (boundaries * 255).astype(np.uint8)
    visualization = cv2.addWeighted(visualization, 0.7, boundaries, 0.3, 0)
    # visualization = boundaries

    # 유효한 페어를 빨간 선으로 연결
    for index1, index2 in valid_pairs:
        row1, col1 = centers[index1]
        row2, col2 = centers[index2]
        cv2.line(visualization, (col1, row1), (col2, row2), color=(0, 0, 255), thickness=2)  # 빨간색 선
    
    cv2.imwrite(gt_path.replace('.png', f'_d3r_{compactness}.png'), visualization)

    return None

if __name__ == "__main__":
    # gt_path = "path_to_ground_truth"
    # est_path = "path_to_estimations"
    # metrics = evaluate_d3r(gt_path, est_path, superpixel_scale=0.5, samples=3000)

    # for metric, values in metrics.items():
    #     print(f"{metric}: {np.nanmean(values):.4f}")
    
    gt_path = os.path.join(os.path.dirname(__file__),'test_disp.png')
    
    # searching_space = [0.001, 0.01, 0.1, 1, 10]
    # for compactness in searching_space:
    #     visualize_d3r(gt_path, superpixel_scale=0.2, samples=5000, compactness=compactness)
    visualize_d3r(gt_path, superpixel_scale=0.2, samples=10000, compactness=0.1)