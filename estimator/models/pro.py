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

import torch
import numpy as np
import torch.nn as nn
from mmengine import print_log
from mmengine.config import ConfigDict
from torchvision.ops import roi_align as torch_roi_align
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig

from estimator.registry import MODELS
from estimator.models import build_model
from estimator.models.baseline_pretrain_disp import BaselinePretrain_disp
from estimator.models.utils import generatemask

from depth_anything.transform import Resize as ResizeDA
from depth_anything.depth_anything_core import DepthAnythingCore_v2
from depth_anything.dpt import DPTHead_disp
from estimator.models.utils import generatemask, generate_relation_dict, divide_images_with_overlap, bboxs_convert_with_overlap_batch, merge_depth_predictions

from estimator.utils import median_norm, min_max_norm_per_sample


@MODELS.register_module()
class PRO(BaselinePretrain_disp, PyTorchModelHubMixin):
    def __init__(
        self, 
        config,):
        nn.Module.__init__(self)
        
        if isinstance(config, ConfigDict):
            # NOTE:
            # convert a MMengine ConfigDict to a huggingface PretrainedConfig
            # it would be used in training and inference without network loading
            config = PretrainedConfig.from_dict(config.to_dict())
            config.load_branch = True
        else:
            # NOTE:
            # used when loading patchfusion from hf model space (inference with network in readme)
            # PretrainedConfig.from_dict(**config) will raise an error (dict is saved as str in this case)
            # we use MMengine ConfigDict to convert str to dict correctly here
            config = PretrainedConfig.from_dict(ConfigDict(**config).to_dict())
            config.load_branch = False
            config.coarse_branch.pretrained_resource = None
            
        self.config = config
        
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        
        self.median = None
        self.mad = None
        
        self.residual = True 
        self.align_corners = True
        self.gpct = getattr(config, 'gpct', False)

        if self.gpct:
            self.image_raw_shape = config.image_raw_shape
            self.overlap = config.overlap
            
            self.rel_dict = generate_relation_dict(config.overlap)
            self.is_loss = build_model(config.is_loss) if getattr(config, 'is_loss', None) else None
        
        self.use_edgeintersect = getattr(config, 'use_edgeintersect', False)
        self.use_unreliable_mask = getattr(config, 'use_unreliable_mask', False)
        
        self.patch_process_shape = config.patch_process_shape
        self.tile_cfg = self.prepare_tile_cfg(config.image_raw_shape, config.patch_split_num)
        
        self.coarse_branch_cfg = config.coarse_branch
        self.coarse_branch = DepthAnythingCore_v2.build(**config.coarse_branch)
    
        self.resizer = ResizeDA(config.patch_process_shape[1], config.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal", align_corners=self.align_corners)
        
        if config.load_branch:
            print_log("Loading coarse_branch from {}".format(config.pretrain_model[0]), logger='current') 
            print_log(self.coarse_branch.core.load_state_dict(torch.load(config.pretrain_model[0], map_location='cpu'), strict=True), logger='current') # coarse ckp
        
        # freeze all these parameters
        for param in self.coarse_branch.parameters():
            param.requires_grad = False
                
        self.loss = build_model(config.loss)
        
        feature_num_dict = {'vits': 64, 'vitb': 128, 'vitl': 256}
        
        config.freq_fusion.align_corners = self.align_corners
        
        self.freq_fusion = build_model(config.freq_fusion)
        
        self.depth_head = DPTHead_disp(features=feature_num_dict[self.config.encoder_type], use_bn=False, concat=config.concat_dpt)
        # NOTE: consistency training
        self.consistency_training = False     
              
    def load_dict(self, dict):
        return self.load_state_dict(dict, strict=False)
                
    def get_save_dict(self):
        current_model_dict = self.state_dict()
        save_state_dict = {}
        for k, v in current_model_dict.items():
            if 'coarse_branch' in k or 'fine_branch' in k:
                pass
            else:
                save_state_dict[k] = v
        return save_state_dict
    
    
    def coarse_forward(self, image_lr):
        with torch.no_grad():
            if self.coarse_branch.training:
                self.coarse_branch.eval()
            
            coarse_prediction, deep_features = self.coarse_branch(image_lr, denorm=False, return_rel_depth=True)        
            
            # Utilize features of DPT Head of coarse branch.
            coarse_features = [
                deep_features[1],
                *deep_features[2:],
                deep_features[0]
            ]

            return coarse_prediction.unsqueeze(1), coarse_features
    
    def coarse_postprocess_train(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):

        coarse_feats_roi = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            cur_lvl_feat = torch_roi_align(feat, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_feats_roi.append(cur_lvl_feat)
        
        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)

        return coarse_prediction_roi, coarse_feats_roi
    

    def coarse_postprocess_test(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):
        patch_num = bboxs_feat.shape[0]

        coarse_feats_roi = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            feat_extend = feat.repeat(patch_num, 1, 1, 1)
            cur_lvl_feat = torch_roi_align(feat_extend, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_feats_roi.append(cur_lvl_feat)
        
        coarse_prediction = coarse_prediction.repeat(patch_num, 1, 1, 1)
        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)

        return_dict = {
            'coarse_depth_roi': coarse_prediction_roi,
            'coarse_feats_roi': coarse_feats_roi}
        
        return return_dict
    
    def fusion_forward(self, fine_depth_pred, crop_input, fine_feats, coarse_depth=None, coarse_depth_roi=None, coarse_feats_roi=None):
        coarse_depth_roi_norm = min_max_norm_per_sample(coarse_depth_roi) 
        fine_depth_pred_norm = min_max_norm_per_sample(fine_depth_pred)

        input_tensor = torch.cat([coarse_depth_roi_norm, fine_depth_pred_norm, crop_input], dim=1)
        
        if self.median is not None and self.mad is not None and self.residual:
            depth_init = (coarse_depth_roi - self.median) / self.mad
        
        depth_predictions = []
        depth_predictions.append(depth_init)
        
        residuals = []
        
        input_tensor = torch.cat([depth_init, input_tensor], dim=1)
        feature_enhanced = self.freq_fusion(
            input_tensor = input_tensor,
            coarse_feats_roi = coarse_feats_roi,
            fine_feats = fine_feats,
        )
        
        residual = self.depth_head(feature_enhanced, self.patch_process_shape[0]/14, self.patch_process_shape[1]/14)
        
        depth = depth_init + residual
        
        #^ depth_predictions[1:]부터 prediction 결과임
        depth_predictions.append(depth)
        residuals.append(residual)
        
        return {'depth_predictions': depth_predictions[1:], 'depth_init':depth_predictions[0], 'residuals': residuals, 'hidden_states': feature_enhanced}

    
    def infer_forward(self, imgs_crop, bbox_feat_forward, tile_temp, coarse_temp_dict):
        
        fine_prediction, fine_features = self.coarse_forward(imgs_crop)
        
        output = \
            self.fusion_forward(
                fine_prediction, 
                imgs_crop, 
                fine_features,
                coarse_depth=tile_temp['coarse_prediction'].repeat(len(imgs_crop), 1, 1, 1),
                **coarse_temp_dict,
                )
            
        depth_predictions = output['depth_predictions']
        return depth_predictions[-1]
    
    def forward(
        self,
        mode,
        image_lr,
        image_hr,
        disp_gt=None,
        crops_image_hr=None,
        crop_disps=None,
        bboxs=None,
        tile_cfg=None,
        cai_mode='m1',
        process_num=4,
        depth_factor = None,
        median_gt=None,
        mad_gt=None,
        mask=None,
        **kwargs
        ):
        if mode == 'train':
            if self.gpct:
                bboxs_feat_factor = torch.tensor([
                    1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                    1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0], 
                    1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                    1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0]], device=bboxs.device).unsqueeze(dim=0)
                patch_raw_shape = kwargs['patch_raw_shape']
                patch_raw_overlap = kwargs['patch_raw_overlap']
                crops_image_hr_list = divide_images_with_overlap(crops_image_hr, self.patch_process_shape, self.overlap)
                #list라고 썼지만 tensor (num_patches, batch_size, 4)임
                bboxs_list = bboxs_convert_with_overlap_batch(bboxs, patch_raw_shape, patch_raw_overlap) 
                
                coarse_prediction, coarse_features = self.coarse_forward(image_lr)
                _, self.median, self.mad = median_norm(coarse_prediction)
                
                if self.use_unreliable_mask:
                    unreliable_mask = kwargs['unreliable_mask_crop']
                    mask = 1 - unreliable_mask
                    
                    if self.use_edgeintersect:
                        gt_edge = kwargs['gt_edge_crop']
                        pred_edge = kwargs['pred_edge_crop']
                        edge_mask = gt_edge * pred_edge
                        
                        mask = (mask.bool() | edge_mask[..., :mask.shape[-2], :mask.shape[-1]].bool()).float()
                
                else:
                    mask = crop_disps > 0
                
                depth_prediction_list = []
                residual_list = []
                for i in range(4):
                    crops_image_hr = crops_image_hr_list[i]
                    bboxs = bboxs_list[i]
                    bboxs_feat = bboxs * bboxs_feat_factor
                    inds = torch.arange(bboxs.shape[0]).to(bboxs.device).unsqueeze(dim=-1)
                    bboxs_feat = torch.cat((inds, bboxs_feat), dim=-1)
                    
                    fine_prediction, fine_features = self.coarse_forward(crops_image_hr)
                    coarse_prediction_roi, coarse_feats_roi = self.coarse_postprocess_train(coarse_prediction, coarse_features, bboxs, bboxs_feat)
                    
                    output = self.fusion_forward(
                        fine_prediction, 
                        crops_image_hr, 
                        fine_features, 
                        coarse_depth= None,
                        coarse_depth_roi=coarse_prediction_roi,
                        coarse_feats_roi=coarse_feats_roi,
                        )
                    
                    depth_prediction = output['depth_predictions'][-1]
                    depth_prediction_list.append(depth_prediction)
                    
                    residual = output['residuals'][-1]
                    residual_list.append(residual)
                    
                depth_prediction = merge_depth_predictions(depth_prediction_list, self.overlap, self.patch_process_shape)
                residual = merge_depth_predictions(residual_list, self.overlap, self.patch_process_shape)
                
                loss_dict = self.loss(depth_prediction, crop_disps, mask, median_gt, mad_gt)
                
                if self.gpct:
                    
                    loss_dict['consistency_loss'] = self.is_loss(depth_prediction_list, self.rel_dict)
                    loss_dict['total_loss'] += loss_dict['consistency_loss']
                                
                return loss_dict, {'rgb': crops_image_hr, 'depth_pred': depth_prediction, 'disp_gt': crop_disps, 'mask' : mask, 'residual': residual}  
                
        else:
            if tile_cfg is None:
                tile_cfg = self.tile_cfg
            else:
                tile_cfg = self.prepare_tile_cfg(tile_cfg['image_raw_shape'], tile_cfg['patch_split_num'])
            
            assert image_hr.shape[0] == 1
            
            coarse_prediction, coarse_features = self.coarse_forward(image_lr)

            _, self.median, self.mad = median_norm(coarse_prediction)
            
            tile_temp = {
                'coarse_prediction': coarse_prediction,
                'coarse_features': coarse_features,}
            
            # In m1 mode, we use a uniform mask.
            if cai_mode == 'm1':
                blur_mask = np.ones((self.patch_process_shape[0], self.patch_process_shape[1])).astype(np.float32)
            else:
                blur_mask = generatemask((self.patch_process_shape[0], self.patch_process_shape[1])) + 1e-3
            # blur_mask = generatemask((self.patch_process_shape[0], self.patch_process_shape[1])) + 1e-3
            blur_mask = torch.tensor(blur_mask, device=image_hr.device)
            avg_depth_map = self.regular_tile(
                offset=[0, 0], 
                offset_process=[0, 0], 
                image_hr=image_hr[0], 
                init_flag=True, 
                tile_temp=tile_temp, 
                blur_mask=blur_mask,
                tile_cfg=tile_cfg,
                process_num=process_num,
                )

            if cai_mode == 'm2' or cai_mode[0] == 'r':
                avg_depth_map = self.regular_tile(
                    offset=[0, tile_cfg['patch_raw_shape'][1]//2], 
                    offset_process=[0, self.patch_process_shape[1]//2], 
                    image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                avg_depth_map = self.regular_tile(
                    offset=[tile_cfg['patch_raw_shape'][0]//2, 0],
                    offset_process=[self.patch_process_shape[0]//2, 0], 
                    image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                avg_depth_map = self.regular_tile(
                    offset=[tile_cfg['patch_raw_shape'][0]//2, tile_cfg['patch_raw_shape'][1]//2],
                    offset_process=[self.patch_process_shape[0]//2, self.patch_process_shape[1]//2], 
                    init_flag=False, image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                
            if cai_mode[0] == 'r':
                blur_mask = generatemask((tile_cfg['patch_raw_shape'][0], tile_cfg['patch_raw_shape'][1])) + 1e-3
                blur_mask = torch.tensor(blur_mask, device=image_hr.device)
                avg_depth_map.resize(tile_cfg['image_raw_shape'])
                patch_num = int(cai_mode[1:]) // process_num
                for i in range(patch_num):
                    avg_depth_map = self.random_tile(
                        image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)

            if cai_mode == 'm1' or cai_mode == 'm2':
                avg_depth_map.resize(tile_cfg['image_raw_shape'])
                
            depth = avg_depth_map.average_map
            depth = depth.unsqueeze(dim=0).unsqueeze(dim=0)
            # depth = self.min_max_norm(depth)

            return depth, {'rgb': image_lr, 'depth_pred': depth, 'disp_gt': disp_gt, 'coarse_prediction': coarse_prediction}