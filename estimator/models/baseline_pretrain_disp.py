# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Zhenyu Li
# Modified by: Byeongjun Kwon, 2025

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from estimator.registry import MODELS
from estimator.models import build_model
from estimator.models.utils import RunningAverageMap


@MODELS.register_module()
class BaselinePretrain_disp(nn.Module):
    def __init__(self, 
                 coarse_branch, 
                 sigloss, 
                 image_raw_shape=(2160, 3840),
                 patch_process_shape=(384, 512),
                 patch_split_num=(4, 4),
                 target='coarse',
                 coarse_branch_zoe=None):
        """ZoeDepth model
        """
        super().__init__()
        
        self.patch_process_shape = patch_process_shape
        self.tile_cfg = self.prepare_tile_cfg(image_raw_shape, patch_split_num)
        
        self.coarse_branch_cfg = coarse_branch
        self.sigloss = build_model(sigloss)
        self.target = target
        

    def prepare_tile_cfg(self, image_raw_shape, patch_split_num):
        # information for process
        patch_split_num = patch_split_num
        patch_reensemble_shape = (self.patch_process_shape[0] * patch_split_num[0], self.patch_process_shape[1] * patch_split_num[1])
        patch_raw_shape = (image_raw_shape[0] // patch_split_num[0], image_raw_shape[1] // patch_split_num[1])
        image_raw_shape = image_raw_shape
        
        raw_h_split_point = []
        raw_w_split_point = []
        for i in range(patch_split_num[0]):
            raw_h_split_point.append(int(patch_raw_shape[0] * i))
        for i in range(patch_split_num[1]):
            raw_w_split_point.append(int(patch_raw_shape[1] * i))
            
        tile_cfg = {
            'patch_split_num': patch_split_num,
            'patch_reensemble_shape': patch_reensemble_shape,
            'patch_raw_shape': patch_raw_shape,
            'image_raw_shape': image_raw_shape,
            'raw_h_split_point': raw_h_split_point,
            'raw_w_split_point': raw_w_split_point}
        
        return tile_cfg
    
    @torch.no_grad()
    def random_tile(
        self, 
        image_hr, 
        tile_temp=None, 
        blur_mask=None, 
        avg_depth_map=None,
        tile_cfg=None,
        process_num=4,):
        ## setting
        height, width = tile_cfg['patch_raw_shape'][0], tile_cfg['patch_raw_shape'][1]

        h_start_list = [random.randint(0, tile_cfg['image_raw_shape'][0] - height - 1) for _ in range(process_num)]
        w_start_list = [random.randint(0, tile_cfg['image_raw_shape'][1] - width - 1)]

        ## prepare data
        imgs_crop = []
        bboxs = []

        for h_start in h_start_list:
            for w_start in w_start_list:
                crop_image = image_hr[:, h_start: h_start+height, w_start: w_start+width]
                crop_image_resized = self.resizer(crop_image.unsqueeze(dim=0)).squeeze(dim=0) # resize to patch_process_shape
                bbox = torch.tensor([w_start, h_start, w_start+width, h_start+height])
                imgs_crop.append(crop_image_resized)
                bboxs.append(bbox)

        imgs_crop = torch.stack(imgs_crop, dim=0)
        bboxs = torch.stack(bboxs, dim=0)

        imgs_crop = imgs_crop.to(image_hr.device)
        bboxs = bboxs.to(image_hr.device).int()
        bboxs_feat_factor = torch.tensor([
                1 / tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                1 / tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0], 
                1 / tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                1 / tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0]], device=bboxs.device).unsqueeze(dim=0)
        bboxs_feat = bboxs * bboxs_feat_factor
        inds = torch.arange(bboxs.shape[0]).to(bboxs.device).unsqueeze(dim=-1)
        bboxs_feat = torch.cat((inds, bboxs_feat), dim=-1)
        
        if tile_temp is not None:
            coarse_postprocess_dict = self.coarse_postprocess_test(bboxs=bboxs, bboxs_feat=bboxs_feat, **tile_temp) 
        
        prediction_list = []
        if tile_temp is not None:
            coarse_temp_dict = {}
            for k, v in coarse_postprocess_dict.items():
                if k == 'coarse_feats_roi':
                    coarse_temp_dict[k] = [f for f in v]
                else:
                    coarse_temp_dict[k] = v
            bbox_feat_forward = bboxs_feat
            bbox_feat_forward[:, 0] = 0
            prediction = self.infer_forward(imgs_crop, bbox_feat_forward, tile_temp, coarse_temp_dict)
        else:
            prediction = self.infer_forward(imgs_crop)
            
        prediction_list.append(prediction)
        predictions = torch.cat(prediction_list, dim=0)
        predictions = F.interpolate(predictions, tile_cfg['patch_raw_shape'])
                  
        patch_select_idx = 0
        for h_start in h_start_list:
            for w_start in w_start_list:
                temp_depth = predictions[patch_select_idx]
                
                count_map = torch.zeros(tile_cfg['image_raw_shape'], device=temp_depth.device)
                pred_depth = torch.zeros(tile_cfg['image_raw_shape'], device=temp_depth.device)
                count_map[h_start: h_start+tile_cfg['patch_raw_shape'][0], w_start: w_start+tile_cfg['patch_raw_shape'][1]] = blur_mask
                pred_depth[h_start: h_start+tile_cfg['patch_raw_shape'][0], w_start: w_start+tile_cfg['patch_raw_shape'][1]] = temp_depth * blur_mask
                avg_depth_map.update(pred_depth, count_map)
                    
                patch_select_idx += 1
        
        return avg_depth_map
    
    
    @torch.no_grad()
    def regular_tile(
        self, 
        offset, 
        offset_process, 
        image_hr, 
        init_flag=False, 
        tile_temp=None, 
        blur_mask=None, 
        avg_depth_map=None,
        tile_cfg=None,
        process_num=4
        ):
        
        ## setting
        height, width = tile_cfg['patch_raw_shape'][0], tile_cfg['patch_raw_shape'][1]
        offset_h, offset_w = offset[0], offset[1]
        assert offset_w >= 0 and offset_h >= 0
        
        tile_num_h = (tile_cfg['image_raw_shape'][0] - offset_h) // height
        tile_num_w = (tile_cfg['image_raw_shape'][1] - offset_w) // width
        h_start_list = [height * h + offset_h for h in range(tile_num_h)]
        w_start_list = [width * w + offset_w for w in range(tile_num_w)]
        
        height_process, width_process = self.patch_process_shape[0], self.patch_process_shape[1]
        offset_h_process, offset_w_process = offset_process[0], offset_process[1]
        assert offset_h_process >= 0 and offset_w_process >= 0
        
        tile_num_h_process = (tile_cfg['patch_reensemble_shape'][0] - offset_h_process) // height_process
        tile_num_w_process = (tile_cfg['patch_reensemble_shape'][1] - offset_w_process) // width_process
        h_start_list_process = [height_process * h + offset_h_process for h in range(tile_num_h_process)]
        w_start_list_process = [width_process * w + offset_w_process for w in range(tile_num_w_process)]
        
        ## prepare data
        imgs_crop = []
        bboxs = []

        for h_start in h_start_list:
            for w_start in w_start_list:
                crop_image = image_hr[:, h_start: h_start+height, w_start: w_start+width]
                crop_image_resized = self.resizer(crop_image.unsqueeze(dim=0)).squeeze(dim=0) # resize to patch_process_shape
                bbox = torch.tensor([w_start, h_start, w_start+width, h_start+height])
                imgs_crop.append(crop_image_resized)
                bboxs.append(bbox)
        
        imgs_crop = torch.stack(imgs_crop, dim=0)
        bboxs = torch.stack(bboxs, dim=0)

        imgs_crop = imgs_crop.to(image_hr.device)
        bboxs = bboxs.to(image_hr.device).int()
        
        bboxs = bboxs.squeeze() # HACK: during inference, 1, 16, 4 -> 16, 4
        if len(bboxs.shape) == 1:
            bboxs = bboxs.unsqueeze(dim=0)
        bboxs_feat_factor = torch.tensor([
                1 / tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                1 / tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0], 
                1 / tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                1 / tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0]], device=bboxs.device).unsqueeze(dim=0)
        bboxs_feat = bboxs * bboxs_feat_factor
        inds = torch.arange(bboxs.shape[0]).to(bboxs.device).unsqueeze(dim=-1)
        # bboxs_feat: [inds, x1, y1, x2, y2]; (16,5)
        bboxs_feat = torch.cat((inds, bboxs_feat), dim=-1)
        
        # post_process
        if tile_temp is not None:
            coarse_postprocess_dict = self.coarse_postprocess_test(bboxs=bboxs, bboxs_feat=bboxs_feat, **tile_temp) 
        
        # count map: counting the number of depth maps that contributed to each pixel
        count_map = torch.zeros(tile_cfg['patch_reensemble_shape'], device=image_hr.device)
        pred_depth = torch.zeros(tile_cfg['patch_reensemble_shape'], device=image_hr.device)
        
        prediction_list = []
        split_rebatch_image = torch.split(imgs_crop, process_num, dim=0)
        for idx, rebatch_image in enumerate(split_rebatch_image):
            if tile_temp is not None:
                coarse_temp_dict = {}
                for k, v in coarse_postprocess_dict.items():
                    if k == 'coarse_feats_roi':
                        coarse_temp_dict[k] = [f[idx*process_num:(idx+1)*process_num, :, :, :] for f in v]
                    else:
                        coarse_temp_dict[k] = v[idx*process_num:(idx+1)*process_num, :, :, :]
                bbox_feat_forward = bboxs_feat[idx*process_num:(idx+1)*process_num, :]
                bbox_feat_forward[:, 0] = 0
                prediction = self.infer_forward(rebatch_image, bbox_feat_forward, tile_temp, coarse_temp_dict)
            else:
                prediction = self.infer_forward(rebatch_image)
            prediction_list.append(prediction)
        predictions = torch.cat(prediction_list, dim=0)
                         
        patch_select_idx = 0
        for h_start in h_start_list_process:
            for w_start in w_start_list_process:
                temp_depth = predictions[patch_select_idx]
                
                if init_flag:
                    count_map[h_start: h_start+self.patch_process_shape[0], w_start: w_start+self.patch_process_shape[1]] = blur_mask
                    pred_depth[h_start: h_start+self.patch_process_shape[0], w_start: w_start+self.patch_process_shape[1]] = temp_depth * blur_mask
            
                else:
                    count_map = torch.zeros(tile_cfg['patch_reensemble_shape'], device=temp_depth.device)
                    pred_depth = torch.zeros(tile_cfg['patch_reensemble_shape'], device=temp_depth.device)
                    count_map[h_start: h_start+self.patch_process_shape[0], w_start: w_start+self.patch_process_shape[1]] = blur_mask
                    pred_depth[h_start: h_start+self.patch_process_shape[0], w_start: w_start+self.patch_process_shape[1]] = temp_depth * blur_mask
                    avg_depth_map.update(pred_depth, count_map)
                    
                patch_select_idx += 1
        
        if init_flag:
            avg_depth_map = RunningAverageMap(pred_depth, count_map, align_corners=self.align_corners)
                        
        return avg_depth_map