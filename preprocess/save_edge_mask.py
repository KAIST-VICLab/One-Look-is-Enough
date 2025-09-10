import os
import os.path as osp
import argparse
import torch
import time
from torch.utils.data import DataLoader
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
import mmengine

from estimator.utils import RunnerInfo, setup_env, fix_random_seed
from estimator.datasets.builder import build_dataset
import cv2

# from depth_anything_original.dpt import DepthAnything
from depth_anything_v2.dpt import DepthAnythingV2
from torchvision.transforms import Normalize
import torch.nn.functional as F
import numpy as np
from estimator.utils.metric import fgbg_depth_thinned
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', 
        help='the dir to save logs and models', 
        default=None)
    parser.add_argument(
        '--test-type',
        type=str,
        default='normal',
        help='evaluation type')
    parser.add_argument(
        '--ckp-path',
        type=str,
        help='ckp_path')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='save colored prediction & depth predictions')
    parser.add_argument(
        '--save_absrel',
        action='store_true',
        default=False,
        help='save absrel vis')
    parser.add_argument(
        '--cai-mode', 
        type=str,
        default='m1',
        help='m1, m2, or rx')
    parser.add_argument(
        '--process-num',
        #! process_num을 2로 설정하면 train할 때의 evaluation 값과 달라져 1로 고정
        type=int, default=1,
        help='batchsize number for inference')
    parser.add_argument(
        '--tag',
        type=str, default='',
        help='infer_infos')
    parser.add_argument(
        '--gray-scale',
        action='store_true',
        default=False,
        help='use gray-scale color map')
    parser.add_argument(
        '--image-raw-shape',
        nargs='+', default=[2160, 3840])
    parser.add_argument(
        '--patch-split-num',
        nargs='+', default=[4, 4])
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def prepare_output_dirs(base_dir, dirs=("pred_edge", "gt_edge", "unreliable_mask")):
    created = []
    for d in dirs:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        created.append(path)
    return created


def save_edge(disp_gt, threshold, save_path, size=None):
    mask_left, mask_top, mask_right, mask_bottom = fgbg_depth_thinned(disp_gt, threshold)
            
    mask_left = np.pad(mask_left.astype(np.uint8), ((0, 0), (0, 1)), mode='constant', constant_values=0)
    mask_top = np.pad(mask_top.astype(np.uint8), ((0, 1), (0, 0)), mode='constant', constant_values=0)
    mask_right = np.pad(mask_right.astype(np.uint8), ((0, 0), (1, 0)), mode='constant', constant_values=0)
    mask_bottom = np.pad(mask_bottom.astype(np.uint8), ((1, 0), (0, 0)), mode='constant', constant_values=0)
    
    edges = mask_left | mask_top | mask_right | mask_bottom
    
    if size is not None:
        edges = cv2.resize(edges, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    # np.save(save_path, edges)
    
    edges = (edges*255).astype(np.uint8)
    cv2.imwrite(save_path, edges)

def save_unreliabe_mask(result, disp_gt, save_path):
    result_norm = (result - result.min()) / (result.max() - result.min()) + 1e-8
    disp_gt_norm = (disp_gt - disp_gt.min()) / (disp_gt.max() - disp_gt.min()) + 1e-8

    result_norm = result_norm.squeeze().cpu().numpy()
    disp_gt_norm = disp_gt_norm.squeeze().cpu().numpy()
    
    ratio = result_norm / disp_gt_norm
    threshold = 2.0
    mask = np.logical_or(ratio > threshold, 1 / ratio > threshold)
    
    # np.save(save_path.replace('.png', '.npy'), mask)
    
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask)

def main():
    args = parse_args()

    image_raw_shape=[int(num) for num in args.image_raw_shape]
    patch_split_num=[int(num) for num in args.patch_split_num]
        
    # load config
    cfg = Config.fromfile(args.config)
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use ckp path as default work_dir if cfg.work_dir is None
        if '.pth' in args.ckp_path:
            args.work_dir = osp.dirname(args.ckp_path)
        else:
            args.work_dir = osp.join('work_dir', args.ckp_path.split('/')[1])
        cfg.work_dir = args.work_dir
        
    mkdir_or_exist(cfg.work_dir)
    cfg.ckp_path = args.ckp_path
    
    # fix seed
    seed = cfg.get('seed', 5621)
    fix_random_seed(seed)
    
    # start dist training
    if cfg.launcher == 'none':
        distributed = False
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
        rank = 0
        world_size = 1
        env_cfg = cfg.get('env_cfg')
    else:
        distributed = True
        env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
        rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)
    
    # build dataloader
    dataloader_config = cfg.val_dataloader
    dataset = build_dataset(cfg.val_dataloader.dataset)
    
    # extract experiment name from cmd
    config_path = args.config
    exp_cfg_filename = config_path.split('/')[-1].split('.')[0]
    # dataset_name = dataset.dataset_name
    dataset_name = getattr(dataset, 'dataset_name', 'middlebury')
    log_filename = 'eval_{}_{}_{}_{}.log'.format(exp_cfg_filename, args.tag, dataset_name, timestamp)
    
    # prepare basic text logger
    log_file = osp.join(args.work_dir, log_filename)
    log_cfg = dict(log_level='INFO', log_file=log_file)
    log_cfg.setdefault('name', timestamp)
    log_cfg.setdefault('logger_name', 'patchstitcher')
    # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    # unexpectedly. Using file mode 'a' can help prevent abnormal
    # termination of the FileHandler and ensure that the log file could
    # be continuously updated during the lifespan of the runner.
    log_cfg.setdefault('file_mode', 'a')
    logger = MMLogger.get_instance(**log_cfg)
    
    # save some information useful during the training
    runner_info = RunnerInfo()
    runner_info.config = cfg # ideally, cfg should not be changed during process. information should be temp saved in runner_info
    runner_info.logger = logger # easier way: use print_log("infos", logger='current')
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.launcher = cfg.launcher
    runner_info.seed = seed
    runner_info.world_size = world_size
    runner_info.work_dir = cfg.work_dir
    runner_info.timestamp = timestamp
    runner_info.save = args.save
    runner_info.save_absrel = args.save_absrel
    runner_info.log_filename = log_filename
    runner_info.gray_scale = args.gray_scale
    
    if runner_info.save:
        mkdir_or_exist(args.work_dir)
        runner_info.work_dir = args.work_dir
    
    
    model_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    model = DepthAnythingV2(**model_config)
    model.load_state_dict(torch.load('/home/bj/Documents/GitHub/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    model = model.eval()
    model.cuda()
    val_sampler = None
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)
    
    prepare_output_dirs(cfg.work_dir, dirs=("pred_edge", "gt_edge", "unreliable_mask"))
    
    threshold = 1.25
    
    # #! pred edge 저장
    prog_bar = mmengine.utils.ProgressBar(len(val_dataloader))
    for idx, batch_data in enumerate(val_dataloader):
        
        image = batch_data['image_lr']
        image = image.cuda()
        image = transform(image)
        with torch.no_grad():
            result = model(image)
            
        result_for_mask = F.interpolate(result.unsqueeze(1), size=batch_data['disp_gt'].shape[-2:], mode='bilinear', align_corners=True)
        result_for_mask = result_for_mask.squeeze(0)
        disp_gt = batch_data['disp_gt']
        
        save_unreliabe_mask(result_for_mask, disp_gt, os.path.join(cfg.work_dir,'unreliable_mask', f'{idx}.png'))
        
        result_norm = (result - result.min()) / (result.max() - result.min())
        save_edge(result_norm.squeeze().cpu().numpy(), threshold, os.path.join(cfg.work_dir, 'pred_edge', f'{idx}.png'), size=batch_data['disp_gt'].shape[-2:])
        save_edge(disp_gt.squeeze().cpu().numpy(), threshold, os.path.join(cfg.work_dir, 'gt_edge', f'{idx}.png'), size=batch_data['disp_gt'].shape[-2:])
        
        prog_bar.update()
    
            
if __name__ == '__main__':
    main()