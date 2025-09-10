import os
import cv2
import numpy as np
import torch
import mmengine
from mmengine.dist import get_dist_info, collect_results_gpu
from mmengine import print_log
from estimator.utils import colorize, median_norm
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset

class Tester_disp:
    """
    Tester class
    """
    def __init__(
        self, 
        config,
        runner_info,
        dataloader,
        model):
       
        self.config = config
        self.runner_info = runner_info
        self.dataloader = dataloader
        self.model = model
        self.collect_input_args = config.collect_input_args
        self.f1 = getattr(dataloader.dataset, 'dataset_name', 'middlebury') == 'u4k'
        dataset_name = getattr(dataloader.dataset, 'dataset_name', 'middlebury')
        self.recall = dataset_name in {'dis5k', 'am2k', 'p3m10k', 'uhrsd'}
        
        self.resize_to_original = getattr(dataloader.dataset, 'dataset_name', 'middlebury') == 'general_res_free' or getattr(dataloader.dataset, 'dataset_name', 'middlebury') == ''
    
    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                if k in self.collect_input_args:
                    collect_batch_data[k] = v.cuda()
        return collect_batch_data
    
    @torch.no_grad()
    def run(self, cai_mode='p16', process_num=4, patch_split_num=[4, 4]):
        
        results = []
        dataset = self.dataloader.dataset
        data_length = len(dataset)
        
        if isinstance(dataset, ConcatDataset):
            dataset = dataset.datasets[0]
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(data_length)

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
            
            batch_data_collect = self.collect_input(batch_data)
            
            tile_cfg = dict()
            raw_h, raw_w = batch_data_collect['image_hr'].shape[-2:]
            tile_cfg['image_raw_shape'] = batch_data_collect['image_hr'].shape[-2:]
            tile_cfg['patch_split_num'] = patch_split_num # use a customized value instead of the default [4, 4] for 4K images
            
            result, log_dict = self.model(mode='infer', cai_mode=cai_mode, process_num=process_num, tile_cfg=tile_cfg, **batch_data_collect) # might use test/val to split cases
            
            assert result.shape[-2:][0] % patch_split_num[0] == 0 and result.shape[-2:][1] % patch_split_num[1] == 0, f"Image size {result.shape[-2:]} should be divisible by patch_split_num {patch_split_num}"
            
            if self.recall:
                if result.shape[-2:] != batch_data_collect['mask'].shape[-2:]:
                    result = F.interpolate(result, size=batch_data_collect['mask'].shape[-2:], mode='bilinear', align_corners=True)
                metrics ={}
                result_norm = (result - result.min()) / (result.max() - result.min())
                recall = dataset.get_recall_score(batch_data_collect['mask'], result_norm.cuda())
                results.append(recall)
            
            # patch_split_num으로 나누어지도록 getitem에서 image 크기가 변경된 경우 depth 출력 값을 원래 크기로 복원
            if batch_data_collect.get('depth_gt', None) is not None:
                if result.shape[-2:] != batch_data_collect['depth_gt'].shape[-2:]:
                    result = F.interpolate(result, size=batch_data_collect['depth_gt'].shape[-2:], mode='bilinear', align_corners=True)
            
            if self.resize_to_original:
                result = F.interpolate(result, size=batch_data['res_original'], mode='bilinear', align_corners=True)
                
            if self.runner_info.save:
                if self.runner_info.save_residual:
                    coarse_prediction = log_dict['coarse_prediction']
                    coarse_prediction, _, _ = median_norm(coarse_prediction)
                    coarse_prediction = F.interpolate(coarse_prediction, size=result.shape[-2:], mode='bilinear', align_corners=True)
                    
                    residual = result - coarse_prediction
                    
                    color_coarse = colorize(coarse_prediction, cmap='Spectral_r', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                    max_res = torch.max(residual.abs()).item()
                    color_res = colorize(residual, vmin=-max_res, vmax=max_res, cmap='seismic')[:, :, [2, 1, 0]]
                    
                    if batch_data.get('img_file_basename', None) is not None:
                        cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_init.png'.format(batch_data['img_file_basename'][0])), color_coarse)
                        cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_res.png'.format(batch_data['img_file_basename'][0])), color_res) 
                    else:
                        cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_init.png'.format(idx)), color_coarse)
                        cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_res.png'.format(idx)), color_res)
                        
                
                if self.runner_info.gray_scale:
                    color_pred = colorize(result, cmap='gray_r')[:, :, [2, 1, 0]]
                else:
                    color_pred = colorize(result, cmap='Spectral_r', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                
                if batch_data.get('img_file_basename', None) is not None:
                    cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}.png'.format(batch_data['img_file_basename'][0])), color_pred)
                else:
                    cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}.png'.format(idx)), color_pred)
            

            if batch_data_collect.get('depth_gt', None) is not None:
                metrics, (_, _) = dataset.get_metrics(
                    batch_data_collect['depth_gt'], 
                    result, 
                    seg_image=batch_data_collect.get('seg_image', None),
                    disp_gt_edges=batch_data.get('boundary', None), 
                    image_hr=batch_data_collect.get('image_hr', None),
                    mask=batch_data_collect.get('mask', None),
                    valid_pairs=batch_data.get('valid_pairs', None),
                    ord_pairs=batch_data.get('ord_pairs', None),
                )
                if self.f1:
                    metrics['f1'] = dataset.get_f1_score(batch_data_collect['disp_gt'], result)

                if metrics is not None:
                    results.extend([metrics])
            
            if self.runner_info.rank == 0:
                batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
        
        if batch_data_collect.get('depth_gt', None) is not None:   
            results = collect_results_gpu(results, data_length)
            if self.runner_info.rank == 0:
                ret_dict = dataset.evaluate(results)
        
        if self.recall:
            results = collect_results_gpu(results, data_length)
            if self.runner_info.rank == 0:
                recall_mean = np.nanmean(results)
                print_log(f"Recall: {recall_mean:.4f}")