_base_ = [
    '../_base_/datasets/u4k_disp_unrel.py', 
    '../_base_/run_time.py'
]

align_corners=True
min_depth=1e-3
max_depth=80
depth_anything_config=dict(
    midas_model_type='vitl',
    img_size=[518, 518],
    pretrained_resource=None,
    use_pretrained_midas=True,
    train_midas=False,
    freeze_midas_bn=True,
    do_resize=False, # do not resize image in midas
    force_keep_ar=True,
    fetch_features=True,
    version = 'v2',
    align_corners=align_corners
)

patch_split_num = (4,4)
overlap=(224,224)
use_edgeintersect=True
use_unreliable_mask=True
gt_kernel_size = (10, 20)
pred_kernel_size = (10, 20)
gpct=True

model=dict(
    type='PRO',
    config=dict(
        encoder_type='vitl',
        image_raw_shape=(2160, 3840),
        patch_split_num=patch_split_num,
        patch_process_shape=(518, 518),
        min_depth=min_depth,
        max_depth=max_depth,
        load_branch=True,
        pretrain_model=['pretrained/Depth-Anything-V2/depth_anything_v2_vitl.pth'],
        coarse_branch=depth_anything_config,
        concat_dpt = True,
        freq_fusion=dict(
            type='Fusion_freq_selective',
            in_channels=[32, 256, 256, 256, 256, 256],
            n_channels=6, 
            wavelet='haar',
            ),
        loss=dict(
            type='AffineInvariantLoss_dict_unrel',
            data_loss=['mae_loss','mse_loss'],
            scales=4,
            align_corners=align_corners,
            loss_weights=dict(
                mae_loss=1.0,
                mse_loss=1.0,
                grad_loss=5.0,
                weight_loss=2.0
                ),
            ),
        is_loss=dict(
            type='ConsistencyLoss',
            data_loss=['mse_loss'],
            loss_weights=dict(
                mae_loss=1.0,
                mse_loss=4.0,
            )
        ),
        align_corners=align_corners,
        overlap=overlap,
        use_edgeintersect=use_edgeintersect,
        use_unreliable_mask=use_unreliable_mask,
        gpct=gpct,
        )
    )
collect_input_args=['image_lr', 'crops_image_hr', 'disp_gt', 'crop_disps', 'bboxs', 'image_hr', 'median_gt', 'mad_gt','depth_gt', 'gt_edge_crop', 'pred_edge_crop', 'unreliable_mask_crop']

project='PRO'

train_cfg=dict(max_epochs=8, val_interval=1, save_checkpoint_interval=2, log_interval=1000, train_log_img_interval=1000, val_log_img_interval=50, val_type='epoch_base', eval_start=0)

optim_wrapper=dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.001),
    clip_grad=dict(type='norm', max_norm=0.1, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
        }),
    accumulative_counts=64
    )

param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=10,
    final_div_factor=10000,
    pct_start=0.25,
    three_phase=False,)

convert_syncbn=True
find_unused_parameters=True

train_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        batch_size=1,
        resize_mode='depth-anything',
        align_corners=align_corners,
        transform_cfg=dict(
            network_process_size=[518, 518],
            patch_split_num=patch_split_num,
            neighbor=True,
            neighbor_shape=(2,2),
            gpct=gpct,
            overlap=overlap,
            ),
        gt_edge_dir= 'Datasets/UnrealStereo4K/BFM/gt_edge',
        pred_edge_dir='Datasets/UnrealStereo4K/BFM/pred_edge',
        unreliable_mask_dir='Datasets/UnrealStereo4K/BFM/unreliable_mask',
        # unreliable_mask_dir='/mnt/Datasets/UnrealStereo4K/mask/unreliable_mask_2.0_v2',
        split='splits/u4k_index/2.0_v2/train.txt'
        ))

val_dataloader=dict(
    dataset=dict(
        resize_mode='depth-anything',
        align_corners=align_corners,
        transform_cfg=dict(
            network_process_size=[518, 518])))

general_dataloader=dict(
    dataset=dict(
        network_process_size=(518, 518),
        align_corners=align_corners,
        resize_mode='depth-anything'))