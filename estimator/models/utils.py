import torch
import cv2
import numpy as np
import torch.nn.functional as F

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class HookTool: 
    def __init__(self):
        self.feat = None 

    def hook_in_fun(self, module, fea_in, fea_out):
        self.feat = fea_in
        
    def hook_out_fun(self, module, fea_in, fea_out):
        self.feat = fea_out

class RunningAverageMap:
    """ Saving avg depth estimation results."""
    def __init__(self, average_map, count_map, align_corners=True):
        self.average_map = average_map
        self.count_map = count_map
        self.average_map = self.average_map / self.count_map
        self.__align_corners = align_corners 

    def update(self, pred_map, ct_map):
        self.average_map = (pred_map + self.count_map * self.average_map) / (self.count_map + ct_map)
        self.count_map = self.count_map + ct_map
        
    def resize(self, resolution):
        temp_avg_map = self.average_map.unsqueeze(0).unsqueeze(0)
        temp_count_map = self.count_map.unsqueeze(0).unsqueeze(0)
        #?align 해야됐는데 안했었다
        self.average_map = F.interpolate(temp_avg_map, size=resolution, mode='bilinear', align_corners=self.__align_corners).squeeze()
        #?align
        self.count_map = F.interpolate(temp_count_map, size=resolution, mode='bilinear', align_corners=self.__align_corners).squeeze()

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.1*size[0]):size[0] - int(0.1*size[0]), int(0.1*size[1]): size[1] - int(0.1*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

def divide_images(image, neighbor_shape):
    # Divide image into patches
    # image: (B, C, H, W)
    # neighbor_shape: (2, 2)
    B, C, H, W = image.size()
    neighbor_shape = torch.tensor(neighbor_shape)
    assert H % neighbor_shape[0] == 0 and W % neighbor_shape[1] == 0
    h, w = H // neighbor_shape[0], W // neighbor_shape[1]
    image = image.view(B, C, neighbor_shape[0], h, neighbor_shape[1], w)
    image = image.permute(2, 4, 0, 1, 3, 5).contiguous()
    image = image.view(neighbor_shape[0] * neighbor_shape[1], B, C, h, w)
    return image

def bboxs_convert(bboxs, neighbor_shape):
    # Input: bboxs (batch_size, 4), neighbor_shape (h_split, w_split)
    # Output: (patch_num, batch_size, 4)

    batch_size = bboxs.shape[0]
    device = bboxs.device
    
    # Initialize a list to store the patch bounding boxes for each batch element
    all_patches = []

    for b in range(batch_size):
        w_start, h_start, w_end, h_end = bboxs[b]
        width = (w_end - w_start) // neighbor_shape[1]
        height = (h_end - h_start) // neighbor_shape[0]

        # Collect patches for the current bounding box
        patch_list = []
        for i in range(neighbor_shape[0]):
            for j in range(neighbor_shape[1]):
                patch_list.append([
                    w_start + j * width,
                    h_start + i * height,
                    w_start + (j + 1) * width,
                    h_start + (i + 1) * height
                ])

        # Append the patches for this batch to the main list
        all_patches.append(torch.tensor(patch_list, device=device))

    # Stack patches along the batch dimension (patch_num, batch_size, 4)
    all_patches = torch.stack(all_patches, dim=1)
    return all_patches

def divide_images_with_overlap(image, patch_shape, overlap):
    """
    Divide image into overlapping patches.
    image: (B, C, H, W) - input image tensor
    patch_shape: (h, w) - shape of each patch
    overlap: (h_overlap, w_overlap) - overlap between patches
    
    Returns:
        # patches: (num_patches, B, C, h, w) - tensor of patches
        patches: list[tnesor(B, C, h, w)] - list of patches
    """
    B, C, H, W = image.size()
    patch_h, patch_w = patch_shape
    stride_h = patch_h - overlap[0]
    stride_w = patch_w - overlap[1]

    # Calculate the number of patches along H and W
    num_patches_h = (H - overlap[0]) // stride_h
    num_patches_w = (W - overlap[1]) // stride_w

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h = i * stride_h
            start_w = j * stride_w
            end_h = start_h + patch_h
            end_w = start_w + patch_w
            patch = image[:, :, start_h:end_h, start_w:end_w]
            patches.append(patch)
    
    # Stack patches into a single tensor
    # patches = torch.stack(patches, dim=0)  # (num_patches, B, C, h, w)
    return patches

def divide_images_with_overlap_batch(image, patch_shape, overlap):
    """
    Divide image into overlapping patches.
    image: (B, C, H, W) - input image tensor
    patch_shape: (B,2) ; (h, w) - shape of each patch
    overlap: (B,2) ; (h_overlap, w_overlap) - overlap between patches
    
    Returns:
        # patches: (num_patches, B, C, h, w) - tensor of patches
        patches: list[tnesor(B, C, h, w)] - list of patches
    """
    B, C, H, W = image.size()
    patch_h, patch_w = patch_shape[:, 0], patch_shape[:, 1]
    overlap_h, overlap_w = overlap[:, 0], overlap[:, 1]
    
    for b in range(B):
        stride_h = patch_h[b] - overlap_h[b]
        stride_w = patch_w[b] - overlap_w[b]

        # Calculate the number of patches along H and W
        num_patches_h = (H - overlap_h[b]) // stride_h
        num_patches_w = (W - overlap_w[b]) // stride_w

        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * stride_h
                start_w = j * stride_w
                end_h = start_h + patch_h[b]
                end_w = start_w + patch_w[b]
                patch = image[:, :, start_h:end_h, start_w:end_w]
                patches.append(patch)

    return patches

def bboxs_convert_with_overlap(bboxs, patch_raw_overlap):
    # Convert bounding boxes to patches with overlap
    # bboxs: (patch_num, batch_size, 4) - bounding boxes for each batch element
    # patch_raw_overlap: (h_overlap, w_overlap) - overlap between patches
    h_overlap, w_overlap = patch_raw_overlap
    overlap_addition = torch.tensor([[0,0,0,0],
                                     [-w_overlap, 0, -w_overlap, 0],
                                     [0, -h_overlap, 0, -h_overlap],
                                     [-w_overlap, -h_overlap, -w_overlap, -h_overlap]], device=bboxs.device) # (4, 4), (patch_num, 4)
    bboxs = bboxs + overlap_addition.unsqueeze(1)
    
    return bboxs

def bboxs_convert_with_overlap_2(bboxs, patch_shape, overlap):
    """
    Convert bounding boxes into overlapping patches.
    bboxs: (batch_size, 4) - input bounding boxes
    patch_shape: (h, w) - shape of each patch
    overlap: (h_overlap, w_overlap) - overlap between patches
    
    Returns:
        overlapping_bboxs: (num_patches, batch_size, 4) - overlapping bounding boxes
        # overlapping_bboxs: list[tensor(batch_size, 4)] - list of overlapping bounding boxes
    """
    batch_size = bboxs.size(0)
    h_overlap, w_overlap = overlap
    patch_h, patch_w = patch_shape

    all_patches = []

    for b in range(batch_size):
        w_start, h_start, w_end, h_end = bboxs[b]
        width = (w_end - w_start - w_overlap) // (patch_w - w_overlap)
        height = (h_end - h_start - h_overlap) // (patch_h - h_overlap)

        patches = []
        for i in range(height):
            for j in range(width):
                patch_w_start = w_start + j * (patch_w - w_overlap)
                patch_h_start = h_start + i * (patch_h - h_overlap)
                patch_w_end = patch_w_start + patch_w
                patch_h_end = patch_h_start + patch_h

                patches.append([patch_w_start, patch_h_start, patch_w_end, patch_h_end])

        all_patches.append(torch.tensor(patches, device=bboxs.device))

    overlapping_bboxs = torch.stack(all_patches, dim=1)  # (num_patches, batch_size, 4)
    return overlapping_bboxs

def bboxs_convert_with_overlap_batch(bboxs, patch_shape, overlap):
    """
    Convert bounding boxes into overlapping patches.
    bboxs: (batch_size, 4) - input bounding boxes
    patch_shape: (batch_size, 2) ; (h, w) - shape of each patch
    overlap: (batch_size, 2) ; (h_overlap, w_overlap)) - overlap between patches
    
    Returns:
        overlapping_bboxs: (num_patches, batch_size, 4) - overlapping bounding boxes
        # overlapping_bboxs: list[tensor(batch_size, 4)] - list of overlapping bounding boxes
    """
    batch_size = bboxs.size(0)
    h_overlap, w_overlap = overlap[:, 0], overlap[:, 1]
    patch_h, patch_w = patch_shape[:, 0], patch_shape[:, 1]

    all_patches = []

    for b in range(batch_size):
        w_start, h_start, w_end, h_end = bboxs[b]
        width = (w_end - w_start - w_overlap[b]) // (patch_w[b] - w_overlap[b])
        height = (h_end - h_start - h_overlap[b]) // (patch_h[b] - h_overlap[b])

        patches = []
        for i in range(height):
            for j in range(width):
                patch_w_start = w_start + j * (patch_w[b] - w_overlap[b])
                patch_h_start = h_start + i * (patch_h[b] - h_overlap[b])
                patch_w_end = patch_w_start + patch_w[b]
                patch_h_end = patch_h_start + patch_h[b]

                patches.append([patch_w_start, patch_h_start, patch_w_end, patch_h_end])

        all_patches.append(torch.tensor(patches, device=bboxs.device))

    overlapping_bboxs = torch.stack(all_patches, dim=1)  # (num_patches, batch_size, 4)
    return overlapping_bboxs


def generate_relation_dict(overlap):
    h_overlap, w_overlap = overlap
    offsets = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1)
    }
    
    relation_dict = {}
    
    for src in range(4):
        relation_dict[str(src)] = {}
        for dst in range(4):
            if src == dst:
                continue
            
            h_src, w_src = offsets[src]
            h_dst, w_dst = offsets[dst]
            
            h_slice = slice(-h_overlap, None) if h_dst > h_src else (slice(None, h_overlap) if h_dst < h_src else slice(None))
            w_slice = slice(-w_overlap, None) if w_dst > w_src else (slice(None, w_overlap) if w_dst < w_src else slice(None))
            
            relation_dict[str(src)][str(dst)] = (h_slice, w_slice)
    
    return relation_dict

def merge_depth_predictions(depth_prediction_list, overlap, patch_size):
    """
    depth_prediction_list: List of 4 tensors of shape (B,1,patch_h,patch_w)
    overlap: (h_overlap, w_overlap) - Overlap size in height and width
    patch_size: (patch_h, patch_w) - Size of individual patches
    full_size: (H, W) - Full image size after merging
    """
    h_overlap, w_overlap = overlap
    h_patch, w_patch = patch_size
    H, W = (h_patch *2 - h_overlap, w_patch * 2 - w_overlap)  # Full image size
    B = depth_prediction_list[0].shape[0]  # Batch size

    # 최종 depth map과 count map 초기화
    final_depth = torch.zeros((B, 1, H, W), device=depth_prediction_list[0].device)
    final_count = torch.zeros_like(final_depth)

    # 패치의 위치 (왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래)
    offsets = {
        0: (0, 0),
        1: (0, w_patch - w_overlap),
        2: (h_patch - h_overlap, 0),
        3: (h_patch - h_overlap, w_patch - w_overlap),
    }

    # 각 패치를 올바른 위치에 배치하고 count 증가
    for i, depth_patch in enumerate(depth_prediction_list):
        h_offset, w_offset = offsets[i]
        final_depth[:, :, h_offset:h_offset + h_patch, w_offset:w_offset + w_patch] += depth_patch
        final_count[:, :, h_offset:h_offset + h_patch, w_offset:w_offset + w_patch] += 1

    # count로 나눠 평균을 계산 (0으로 나누는 걸 방지)
    final_depth /= final_count.clamp(min=1)

    return final_depth
