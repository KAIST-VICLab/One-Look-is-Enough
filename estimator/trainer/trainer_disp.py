import os
import wandb
import numpy as np
import torch
import mmengine
from mmengine.optim import build_optim_wrapper
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.distributed as dist
from mmengine.dist import get_dist_info, collect_results_gpu
from mmengine import print_log
import torch.nn.functional as F
from tqdm import tqdm
from estimator.utils import colorize
import math
from estimator.utils import median_norm, AsymmetricDilation

class Trainer_disp:
    """
    Trainer class
    """
    def __init__(
        self, 
        config,
        runner_info,
        train_sampler,
        train_dataloader,
        val_dataloader,
        model,
        additional_val_dataloader=None,
        ):
       
        self.config = config
        self.runner_info = runner_info
        
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.additional_val_dataloader = additional_val_dataloader
        self.model = model
        
        self.gpct = getattr(config, 'gpct', False) 
        self.neighbor = getattr(config.train_dataloader.dataset.transform_cfg, 'neighbor', False)
        if self.neighbor:
            self.neighbor_shape = getattr(config.train_dataloader.dataset.transform_cfg, 'neighbor_shape', (2, 2))
            
            if self.gpct:
                self.patch_process_shape = config.model.config.patch_process_shape
                self.image_raw_shape = config.model.config.image_raw_shape
                self.overlap = config.model.config.overlap
        
        # build opt and schedule
        self.optimizer_wrapper = build_optim_wrapper(self.model, config.optim_wrapper)
        
        self.accumulative_counts = self.config.optim_wrapper.get('accumulative_counts', 1)
        if self.neighbor and not self.gpct:
            steps_per_epoch = math.ceil(len(self.train_dataloader) / self.accumulative_counts * self.neighbor_shape[0] * self.neighbor_shape[1])
        else:
            steps_per_epoch =math.ceil(len(self.train_dataloader) / self.accumulative_counts)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer_wrapper.optimizer, [l['lr'] for l in self.optimizer_wrapper.optimizer.param_groups], epochs=self.config.train_cfg.max_epochs, steps_per_epoch=steps_per_epoch,
            cycle_momentum=config.param_scheduler.cycle_momentum, base_momentum=config.param_scheduler.get('base_momentum', 0.85), max_momentum=config.param_scheduler.get('max_momentum', 0.95),
            div_factor=config.param_scheduler.div_factor, final_div_factor=config.param_scheduler.final_div_factor, pct_start=config.param_scheduler.pct_start, three_phase=config.param_scheduler.three_phase)
    
        # I'd like use wandb log_name
        self.train_step = 0 # for training
        self.val_step = 0 # for validation

        self.iters_per_train_epoch = len(self.train_dataloader)
        self.iters_per_val_epoch = len(self.val_dataloader)
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.collect_input_args = config.collect_input_args
        self.collect_input_args.append('mask')
        print_log('successfully init trainer', logger='current')
        
        self.use_edge_relmask = False
        if self.train_dataloader.dataset.dataset_name == 'u4k_unrel':
            self.use_edge_relmask = True
            self.dilate_gt = AsymmetricDilation(config.gt_kernel_size[0], config.gt_kernel_size[1])
            self.dilate_pred = AsymmetricDilation(config.pred_kernel_size[0], config.pred_kernel_size[1])
            
            self.dilate_gt = self.dilate_gt.cuda(runner_info.rank)
            self.dilate_pred = self.dilate_pred.cuda(runner_info.rank)
    
    
    def log_images(self, log_dict, prefix="", scalar_cmap="turbo", min_depth=1e-3, max_depth=80, step=0):
        # Custom log images. Please add more items to the log dict returned from the model
        
        wimages = dict()
        wimages['{}/step'.format(prefix)] = step
        
        rgb = log_dict.get('rgb')[0]
        _, h_rgb, w_rgb = rgb.shape
        
        # save disp
        if 'depth_pred' in log_dict.keys():
            depth_pred = log_dict.get('depth_pred')[0]
            depth_pred = depth_pred.squeeze()
            depth_gt = log_dict.get('disp_gt')[0]
            depth_gt = depth_gt.squeeze()
            
            depth_gt_color = colorize(depth_gt, vmin=None, vmax=None, cmap=scalar_cmap)
            depth_pred_color = colorize(depth_pred, vmin=None, vmax=None, cmap=scalar_cmap)
            
            depth_gt_img = wandb.Image(depth_gt_color, caption='disp_gt')
            depth_pred_img = wandb.Image(depth_pred_color, caption='disp_pred')
            rgb = wandb.Image(rgb, caption='rgb')
            
            wimages['{}/LogImageDepth'.format(prefix)] = [rgb, depth_gt_img, depth_pred_img]
        
        if 'depth_init' in log_dict.keys() and log_dict.get('depth_init') is not None:
            depth_init = log_dict.get('depth_init')[0]
            depth_init = depth_init.squeeze()
            depth_init_color = colorize(depth_init, vmin=None, vmax=None, cmap=scalar_cmap)
            depth_init_img = wandb.Image(depth_init_color, caption='disp_init')
            cur_log = wimages['{}/LogImageDepth'.format(prefix)]
            cur_log.append(depth_init_img)
            wimages['{}/LogImageDepth'.format(prefix)] = cur_log
        
        if 'residual' in log_dict.keys() and log_dict.get('residual') is not None:
            residual = log_dict.get('residual')[0]
            residual = residual.squeeze()
            max = torch.max(residual.abs()).item()
            residual_color = colorize(residual, vmin=-max, vmax=max, cmap='seismic')
            residual_img = wandb.Image(residual_color, caption='resiudal')
            
            wimages['{}/LogImageResiduals'.format(prefix)] = residual_img
        
        if 'coarse_prediction' in log_dict.keys() and log_dict.get('coarse_prediction') is not None:
            coarse_prediction = log_dict.get('coarse_prediction')[0]
            depth_pred = log_dict.get('depth_pred')[0]
            coarse_prediction, _, _ = median_norm(coarse_prediction)
            coarse_prediction = F.interpolate(coarse_prediction.unsqueeze(0), size=depth_pred.shape[-2:], mode='bilinear', align_corners=True).squeeze(0)
            residual = depth_pred - coarse_prediction
            
            max_res = torch.max(residual.abs()).item()
            color_res = colorize(residual, vmin=-max_res, vmax=max_res, cmap='seismic')
            res_img = wandb.Image(color_res, caption='residual')
            cur_log = wimages['{}/LogImageDepth'.format(prefix)]
            cur_log.append(res_img)
            wimages['{}/LogImageDepth'.format(prefix)] = cur_log
            
        if 'mask' in log_dict.keys():
            mask = log_dict.get('mask')[0]
            mask = mask.squeeze().float()*255
            mask_img = wandb.Image(
                mask.unsqueeze(-1).detach().cpu().numpy(),
                caption='valid_mask')
            cur_log = wimages['{}/LogImageDepth'.format(prefix)]
            cur_log.append(mask_img)
            wimages['{}/LogImageDepth'.format(prefix)] = cur_log
        
        wandb.log(wimages)
            

    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                if k in self.collect_input_args:
                    collect_batch_data[k] = v.cuda()
        return collect_batch_data
                    
                    
    @torch.no_grad()
    def val_epoch(self):
        results = []
        results_list = [[] for _ in range(8)]
        
        self.model.eval()
        dataset = self.val_dataloader.dataset
        loader_indices = self.val_dataloader.batch_sampler

        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.val_dataloader)):
            self.val_step += 1

            batch_data_collect = self.collect_input(batch_data)
            # result, log_dict = self.model(mode='infer',  **batch_data_collect)
            result, log_dict = self.model(mode='infer', cai_mode='m1', process_num=1, **batch_data_collect) # might use test/val to split cases

            if isinstance(result, list):
                # in case you have multiple results
                for num_res in range(len(result)):
                    metrics, _ = dataset.get_metrics(
                        batch_data_collect['depth_gt'], 
                        result[num_res], 
                        disp_gt_edges=batch_data.get('boundary', None), 
                        additional_mask=log_dict.get('mask', None),
                        image_hr=batch_data.get('image_hr', None))
                    metrics['f1'] = dataset.get_f1_score(batch_data_collect['disp_gt'], result)
                    results_list[num_res].extend([metrics])
            
            else:
                metrics, _ = dataset.get_metrics(
                    batch_data_collect['depth_gt'], 
                    result, 
                    seg_image=batch_data_collect.get('seg_image', None),
                    disp_gt_edges=batch_data.get('boundary', None), 
                    additional_mask=log_dict.get('mask', None), 
                    image_hr=batch_data.get('image_hr', None))
                metrics['f1'] = dataset.get_f1_score(batch_data_collect['disp_gt'], result)
                results.extend([metrics])

            if self.runner_info.rank == 0:
                if isinstance(result, list):
                    batch_size = len(result[0]) * world_size
                else:
                    batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()

            if self.runner_info.rank == 0 and self.config.debug == False and (idx + 1) % int(self.config.train_cfg.val_log_img_interval / batch_size) == False:
                self.log_images(log_dict=log_dict, prefix="Val", min_depth=self.config.min_depth, max_depth=self.config.max_depth, step=self.val_step)
            
        # collect results from all ranks
        if isinstance(result, list):
            results_collect = []
            for results in results_list:
                results = collect_results_gpu(results, len(dataset))
                results_collect.append(results)
        else:
            results = collect_results_gpu(results, len(dataset))
            
        if self.runner_info.rank == 0:
            if isinstance(result, list):
                for num_refine in range(len(result)):
                    ret_dict = dataset.evaluate(results_collect[num_refine])
            else:
                ret_dict = dataset.evaluate(results)

        if self.runner_info.rank == 0 and self.config.debug == False:
            wdict = dict()
            for k, v in ret_dict.items():
                wdict["Val/{}".format(k)] = v.item()
            wdict['Val/step'] = self.val_step
            wandb.log(wdict)
        
        torch.cuda.empty_cache()
        if self.runner_info.distributed is True:
            torch.distributed.barrier()
        
        self.model.train() # avoid changing model state 
    
    def train_epoch(self, epoch_idx):
        self.model.train()
        if self.runner_info.distributed:
            dist.barrier()

        running_loss = 0.0
        pbar = tqdm(enumerate(self.train_dataloader),
                    desc=f"Epoch: [{epoch_idx + 1}/{self.config.train_cfg.max_epochs}]. Loop: Train",
                    total=self.iters_per_train_epoch) if self.runner_info.rank == 0 else enumerate(self.train_dataloader)

        for idx, batch_data in pbar:
            batch_data_collect = self.collect_input(batch_data)

            if self.gpct:
                patch_raw_shape = batch_data['patch_raw_shape']
                patch_raw_overlap = batch_data['patch_raw_overlap']

                if self.use_edge_relmask:
                    batch_data_collect['gt_edge_crop'] = self.dilate_gt(batch_data_collect['gt_edge_crop'])
                    batch_data_collect['pred_edge_crop'] = self.dilate_pred(batch_data_collect['pred_edge_crop'])

                with self.optimizer_wrapper.optim_context(self.model):
                    loss_dict, log_dict = self.model(mode='train', **batch_data_collect,
                                                    patch_raw_shape=patch_raw_shape,
                                                    patch_raw_overlap=patch_raw_overlap)

                total_loss = loss_dict['total_loss']
                running_loss += total_loss.item()
                self.optimizer_wrapper.update_params(total_loss)

                # Step when accumulation count met
                if self.optimizer_wrapper.should_update():
                    self.scheduler.step()
                    self.train_step += 1
                    running_loss_save = running_loss / self.accumulative_counts
                    running_loss = 0.0

                    # Logging (only on rank 0)
                    if self.runner_info.rank == 0:
                        log_info = f'Epoch: [{epoch_idx + 1:02d}/{self.config.train_cfg.max_epochs:02d}] - Step: [{self.train_step:05d}] - Total Loss: {running_loss_save:.4f}'
                        for k, v in loss_dict.items():
                            if k != 'total_loss':
                                log_info += f' - {k}: {v.item():.3f}' if isinstance(v, torch.Tensor) else f' - {k}: {v:.3f}'
                        print_log(log_info, logger='current')

                        if not self.config.debug:
                            wdict = {
                                'Train/total_loss': running_loss_save,
                                'Train/LR': self.optimizer_wrapper.get_lr()['lr'][0],
                                'Train/momentum': self.optimizer_wrapper.get_momentum()['momentum'][0],
                                'Train/step': self.train_step
                            }
                            for k, v in loss_dict.items():
                                if k != 'total_loss':
                                    wdict[f'Train/{k}'] = v.item() if isinstance(v, torch.Tensor) else v
                            wandb.log(wdict)

                            # Log images periodically
                            if self.config.train_cfg.train_log_img_interval and \
                            self.train_step % self.config.train_cfg.train_log_img_interval == 0:
                                self.log_images(log_dict=log_dict,
                                                prefix="Train",
                                                min_depth=self.config.min_depth,
                                                max_depth=self.config.max_depth,
                                                step=self.train_step)

                    # Iter-based validation
                    if self.config.train_cfg.val_type == 'iter_base' and \
                    (self.train_step % self.config.train_cfg.val_interval == 0) and \
                    (self.train_step >= self.config.train_cfg.get('eval_start', 0)):
                        self.val_epoch()

            # Optional: Update progress bar description every iteration
            if self.runner_info.rank == 0 and 'loss_dict' in locals():
                log_desc = f"Epoch: [{epoch_idx + 1:02d}/{self.config.train_cfg.max_epochs}]"
                for k, v in loss_dict.items():
                    log_desc += f" - {k}: {v.item():.2f}"
                pbar.set_description(log_desc)

                
    def save_checkpoint(self, epoch_idx):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # As default, the model is wrappered by DDP!!! Hence, even if you're using one gpu, please use dist_train.sh
            if hasattr(self.model.module, 'get_save_dict'):
                print_log('Saving ckp, but use the inner get_save_dict fuction to get model_dict', logger='current')
                # print_log('For saving space. Would you like to save base model several times? :>', logger='current')
                model_dict = self.model.module.get_save_dict()
            else:
                model_dict = self.model.module.state_dict() 
        else:
            if hasattr(self.model, 'get_save_dict'):
                print_log('Saving ckp, but use the inner get_save_dict fuction to get model_dict', logger='current')
                # print_log('For saving space. Would you like to save base model several times? :>', logger='current')
                model_dict = self.model.get_save_dict()
            else:
                model_dict = self.model.state_dict()
            
        checkpoint_dict = {
            'epoch': epoch_idx, 
            'model_state_dict': model_dict, 
            'optim_state_dict': self.optimizer_wrapper.state_dict(),
            'schedule_state_dict': self.scheduler.state_dict()}
        
        if self.runner_info.rank == 0:
            torch.save(checkpoint_dict, os.path.join(self.runner_info.work_dir, 'checkpoint_{:02d}.pth'.format(epoch_idx + 1)))
        log_info = 'save checkpoint_{:02d}.pth at {}'.format(epoch_idx + 1, self.runner_info.work_dir)
        print_log(log_info, logger='current')
    
    def resume_checkpoint(self, checkpoint_path):
        """
        Resume training from a saved checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.runner_info.rank))
        epoch_idx = checkpoint['epoch']

        # Load model state dict
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Load optimizer state
        self.optimizer_wrapper.load_state_dict(checkpoint['optim_state_dict'])

        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['schedule_state_dict'])

        print_log(f"Resumed training from checkpoint at epoch {epoch_idx + 1}", logger='current')
        return epoch_idx + 1
    
    def run(self, resume_path=None):
        """
        Main training loop.
        Args:
            resume_path (str, optional): Path to the checkpoint to resume from.
        """
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_path is not None:
            start_epoch = self.resume_checkpoint(resume_path)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print_log(f"Training param: {name}", logger='current')

        for epoch_idx in range(start_epoch, self.config.train_cfg.max_epochs):
            if self.runner_info.distributed:
                self.train_sampler.set_epoch(epoch_idx)
            self.train_epoch(epoch_idx)

            if (
                (epoch_idx + 1) % self.config.train_cfg.val_interval == 0
                and (epoch_idx + 1) >= self.config.train_cfg.get('eval_start', 0)
                and self.config.train_cfg.val_type == 'epoch_base'
            ):
                self.val_epoch()

            if (epoch_idx + 1) % self.config.train_cfg.save_checkpoint_interval == 0:
                self.save_checkpoint(epoch_idx)

            if (epoch_idx + 1) % self.config.train_cfg.get('early_stop_epoch', 9999999) == 0:
                print_log(f"Early stop at epoch: {epoch_idx}", logger='current')
                break

        if self.config.train_cfg.val_type == 'iter_base':
            self.val_epoch()