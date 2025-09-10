import torch
import random
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
from zoedepth.models.base_models.midas import Resize
from depth_anything.transform import Resize as ResizeDA
import os.path as osp
from collections import OrderedDict
from prettytable import PrettyTable
from mmengine import print_log
import copy
from estimator.datasets.transformers import aug_color, aug_flip, to_tensor, random_crop, aug_rotate
from estimator.registry import DATASETS
from estimator.utils import get_boundaries, compute_metrics, compute_depth_from_disp, compute_aligned_disp, SI_boundary_F1
import cv2


@DATASETS.register_module()
class UnrealStereo4kDataset_disp_unrel(Dataset):
    def __init__(
        self,
        mode,
        data_root, 
        split,
        transform_cfg,
        min_depth,
        max_depth,
        resize_mode='zoe',
        align_corners=True,
        batch_size=2,
        gt_edge_dir = '/mnt_2/Datasets/UnrealStereo4K/edges/disp/disp_edge_train_pro_125',
        pred_edge_dir =  '/mnt_2/Depth_anything_v2_results/examples_4k/edges/high/1.25',
        unreliable_mask_dir =  '/mnt_2/Depth_anything_v2_results/examples_4k/unreliable_mask_1.5'
        ):
        
        self.dataset_name = 'u4k_unrel'
        
        self.mode = mode
        self.data_root = data_root
        self.split = split
        self.gt_edge_dir = gt_edge_dir
        self.pred_edge_dir = pred_edge_dir
        self.unreliable_mask_dir = unreliable_mask_dir
        self.data_infos = self.load_data_list()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.batch_size = batch_size
        
        # load transform info
        # not for zoedepth (also da-zoe): do resize, but no normalization. Consider putting normalization in model forward now
        net_h, net_w = transform_cfg.network_process_size
        self.net_h, self.net_w = net_h, net_w
        if resize_mode == 'zoe':
            self.resize = Resize(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
            self.normalize = None
        elif resize_mode == 'depth-anything':
            self.resize = ResizeDA(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal", align_corners=align_corners)
            self.normalize = None
        else:
            raise NotImplementedError
            
        # self.patch_raw_shape = patch_raw_shape
        # transform_cfg.random_crop_size = patch_raw_shape
        self.transform_cfg = transform_cfg
        
        self.patch_split_num = getattr(self.transform_cfg, 'patch_split_num', (4, 4))
        
        self.neighbor = getattr(self.transform_cfg, 'neighbor', False)
        self.gpct = getattr(self.transform_cfg, 'gpct', False)
        if self.neighbor:
            self.neighbor_shape = getattr(self.transform_cfg, 'neighbor_shape', (2,2))
            
            self.overlap = getattr(self.transform_cfg, 'overlap', (0,0)) if self.gpct else (0,0)
            # assert self.overlap[0] % 14 == 0 and self.overlap[1] % 14 == 0, 'overlap should be divisible by 14'
            if resize_mode == 'zoe':
                self.resize_neighbor = Resize(net_w * self.neighbor_shape[1]-self.overlap[1], net_h * self.neighbor_shape[0]-self.overlap[0], keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
            elif resize_mode == 'depth-anything':
                self.resize_neighbor = ResizeDA(net_w * self.neighbor_shape[1]-self.overlap[1], net_h * self.neighbor_shape[0]-self.overlap[0], keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal", align_corners=align_corners)
            else:
                raise NotImplementedError
            
        self.max_split = getattr(self.transform_cfg, 'max_split', 4)
        
        assert 2160 % self.patch_split_num[0] == 0 and 3840 % self.patch_split_num[1] == 0, '(2160, 3840) should be divisible by patch_split_num'
        
        
    def load_data_list(self):
        """Load annotation from directory.
        Args:
            data_root (str): Data root for img_dir/ann_dir.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        data_root = self.data_root
        split = self.split
        
        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info_l = dict()
                    
                    img_l, img_r, depth_map_l, depth_map_r, index = line.strip().split(" ")

                    # HACK: a hack to replace the png with raw to accelerate training
                    img_l = img_l[:-3] + 'raw'

                    img_info_l['depth_map_path'] = osp.join(data_root, depth_map_l)
                    img_info_l['img_path'] = osp.join(data_root, img_l)
                    img_info_l['filename'] = img_l

                    img_info_l['depth_fields'] = []
                    filename = img_info_l['depth_map_path']
                    ext_name_l = filename.replace('Disp0', 'Extrinsics0')
                    ext_name_l = ext_name_l.replace('npy', 'txt')
                    ext_name_r = filename.replace('Disp0', 'Extrinsics1')
                    ext_name_r = ext_name_r.replace('npy', 'txt')
                    with open(ext_name_l, 'r') as f:
                        ext_l = f.readlines()
                    with open(ext_name_r, 'r') as f:
                        ext_r = f.readlines()
                    f = float(ext_l[0].split(' ')[0])
                    img_info_l['focal'] = f
                    base = abs(float(ext_l[1].split(' ')[3]) - float(ext_r[1].split(' ')[3]))
                    img_info_l['depth_factor'] = base * f
                    
                    if self.mode == 'train':
                        index_png = index + '.png'
                        img_info_l['gt_edge_path'] = osp.join(self.gt_edge_dir, index_png)
                        img_info_l['pred_edge_path'] = osp.join(self.pred_edge_dir, index_png)
                        img_info_l['unreliable_mask_path'] = osp.join(self.unreliable_mask_dir, index_png)
                        
                    else:
                        raise NotImplementedError

                    img_infos.append(img_info_l)
        else:
            raise NotImplementedError 

        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        return img_infos
    

    def __getitem__(self, idx):
        img_file_path = self.data_infos[idx]['img_path']
        disp_path = self.data_infos[idx]['depth_map_path']
        depth_factor = self.data_infos[idx]['depth_factor']

        image = np.fromfile(open(img_file_path, 'rb'), dtype=np.uint8).reshape(2160, 3840, 3) 
        
        # load depth
        disp_gt = np.load(disp_path, mmap_mode='c').astype(np.float32)
        depth_gt = depth_factor / disp_gt
        
        
        gt_edge = (cv2.imread(self.data_infos[idx]['gt_edge_path'], cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)
        pred_edge = (cv2.imread(self.data_infos[idx]['pred_edge_path'], cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)
        unreliable_mask = cv2.imread(self.data_infos[idx]['unreliable_mask_path'], cv2.IMREAD_GRAYSCALE) / 255.0
        
        gt_edge_tensor = to_tensor(gt_edge)
        pred_edge_tensor = to_tensor(pred_edge)
        unreliable_mask_tensor = to_tensor(unreliable_mask)
        
        # convert to rgb, it's only for u4k
        image = image.astype(np.float32)[:, :, ::-1].copy()
        
        # div 255
        image = image / 255.0
        
        if self.mode == 'train':
            image = aug_color(image)
            image, gt_info = aug_flip(image, [depth_gt, disp_gt, gt_edge_tensor, pred_edge_tensor, unreliable_mask_tensor])
            depth_gt, disp_gt, gt_edge_tensor, pred_edge_tensor, unreliable_mask_tensor = gt_info[0], gt_info[1], gt_info[2], gt_info[3], gt_info[4]
        
        # process for the coarse input
        image_tensor = to_tensor(image)
        if self.normalize is not None:
            image_tensor = self.normalize(image_tensor) # feed into light branch (hr)
            
        # image_hr_tensor = copy.deepcopy(image_tensor)
        image_lr_tensor = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0) # feed into deep branch (lr, zoe)
        depth_gt_tensor = to_tensor(depth_gt)
        
        img_file_basename, _ = osp.splitext(self.data_infos[idx]['filename'])
        img_file_basename = img_file_basename.replace('/', '_')[1:]
        
        
        if self.mode == 'train':
            disp_gt_tensor = to_tensor(disp_gt)
            median = torch.median(disp_gt_tensor)
            mad = torch.mean(torch.abs(disp_gt_tensor - median))
            
            if self.neighbor:
                if self.gpct:
                    self.patch_raw_shape = (2160//self.patch_split_num[0], 3840//self.patch_split_num[1])
                    self.patch_raw_overlap = (round(self.patch_raw_shape[0]/self.net_h*self.overlap[0]), round(self.patch_raw_shape[1]/self.net_w*self.overlap[1]))
                    
                h_count, w_count = self.neighbor_shape
                self.transform_cfg.random_crop_size = (int(2160/self.patch_split_num[0]*h_count), int(3840/self.patch_split_num[1]*w_count))
                if self.gpct:
                    self.transform_cfg.random_crop_size = (self.transform_cfg.random_crop_size[0]-self.patch_raw_overlap[0], self.transform_cfg.random_crop_size[1]-self.patch_raw_overlap[1])
                h, w = self.transform_cfg.random_crop_size
                image_tensor, gt_info, crop_info = random_crop(image_tensor, [disp_gt_tensor, gt_edge_tensor, pred_edge_tensor, unreliable_mask_tensor], self.transform_cfg.random_crop_size)
                disp_gt_crop_tensor = gt_info[0]
                
                crop_images = self.resize_neighbor(image_tensor.unsqueeze(dim=0)).squeeze(dim=0)
                crop_disps = disp_gt_crop_tensor
                bboxs = torch.tensor([crop_info[1], crop_info[0], crop_info[1]+w, crop_info[0]+h])
                
                return_dict = \
                    {'image_lr': image_lr_tensor, 
                    'image_hr': torch.tensor([2160, 3840]), # save some memory
                    'crops_image_hr': crop_images, 
                    'disp_gt': disp_gt_tensor, 
                    'crop_disps': crop_disps,
                    'bboxs': bboxs,
                    'img_file_basename': img_file_basename,
                    'depth_factor': depth_factor,
                    'median_gt': median,
                    'mad_gt': mad,
                    'gt_edge_crop': gt_info[1],
                    'pred_edge_crop': gt_info[2],
                    'unreliable_mask_crop': gt_info[3]
                    }
                
                if self.gpct:
                    return_dict.update({'patch_raw_shape': torch.tensor(self.patch_raw_shape), 
                                        'patch_raw_overlap': torch.tensor(self.patch_raw_overlap)})
            
        else:
            
            boundary = get_boundaries(disp_gt, th=1, dilation=0) # for eval maybe
            boundary_tensor = to_tensor(boundary)
    
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': image_tensor, 
                 'depth_gt': depth_gt_tensor, 
                 'disp_gt': to_tensor(disp_gt),
                 'boundary': boundary_tensor, 
                 'img_file_basename': img_file_basename,
                 }
                
        return return_dict
            

    def __len__(self):
        # return len(self.data_infos[:10])
        return len(self.data_infos)
    
    def get_metrics(self, depth_gt, result, disp_gt_edges, **kwargs):
        depth_gt = depth_gt.squeeze(1)
        #! scale, shift를 구할 때에 전체 결과 값에서 구할 것인지 gt_depth가 80보다 작은 부분에서 구할 것인지 아직 확실치 않음
        mask = torch.ones_like(depth_gt, dtype=depth_gt.dtype)
        # mask = (depth_gt < 80).to(depth_gt.dtype)
        result = compute_depth_from_disp(result.squeeze(1), depth_gt, mask=mask)
        return compute_metrics(depth_gt, result, disp_gt_edges=disp_gt_edges, min_depth_eval=self.min_depth, max_depth_eval=self.max_depth, garg_crop=False, eigen_crop=False, dataset='')
    
    def get_f1_score(self, disp_gt, result):
        disp_aligned = compute_aligned_disp(result.squeeze(1), disp_gt.squeeze(1))
        disp_aligned = np.array(disp_aligned.squeeze().cpu())
        disp_gt = np.array(disp_gt.squeeze().cpu())
        f1_score = SI_boundary_F1(disp_aligned, disp_gt)

        return f1_score
    
    def pre_eval_to_metrics(self, pre_eval_results):
        aggregate = []
        for item in pre_eval_results:
            aggregate.append(item.values())
        pre_eval_results = aggregate
            
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        pre_eval_results = tuple(zip(*pre_eval_results))
        ret_metrics = OrderedDict({})

        ret_metrics['a1'] = np.nanmean(pre_eval_results[0])
        ret_metrics['a2'] = np.nanmean(pre_eval_results[1])
        ret_metrics['a3'] = np.nanmean(pre_eval_results[2])
        ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[3])
        ret_metrics['rmse'] = np.nanmean(pre_eval_results[4])
        ret_metrics['log_10'] = np.nanmean(pre_eval_results[5])
        ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[6])
        ret_metrics['silog'] = np.nanmean(pre_eval_results[7])
        ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[8])
        ret_metrics['see'] = np.nanmean(pre_eval_results[9])
        ret_metrics['f1'] = np.nanmean(pre_eval_results[10])

        ret_metrics = {metric: value for metric, value in ret_metrics.items()}

        return ret_metrics

    def evaluate(self, results, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict depth map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        
        eval_results = {}
        # test a list of files
        ret_metrics = self.pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 11
        for i in range(num_table):
            names = ret_metric_names[i*11: i*11 + 11]
            values = ret_metric_values[i*11: i*11 + 11]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 7)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Evaluation Summary: \n' + summary_table_data.get_string(), logger='current')

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results