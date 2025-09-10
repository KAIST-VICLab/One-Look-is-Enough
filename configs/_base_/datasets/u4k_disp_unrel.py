

train_dataloader=dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset_disp_unrel',
        mode='train',
        data_root='/mnt/Datasets/UnrealStereo4K',
        split='splits/u4k_index/2.0_v2/train.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            degree=1.0,
            random_crop=True, # random_crop_size will be set as patch_raw_shape
            network_process_size=[384, 512]),
        gt_edge_dir='/mnt/Datasets/UnrealStereo4K/edges/disp/disp_edge_train_pro_125',
        pred_edge_dir='/mnt/Datasets/UnrealStereo4K/edges/pred/1.25',
        unreliable_mask_dir='/mnt/Datasets/UnrealStereo4K/mask/unreliable_mask_1.5',
        ))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset_disp',
        mode='infer',
        data_root='/mnt/Datasets/UnrealStereo4K',
        split='splits/u4k/val.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))

test_in_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset_disp',
        mode='infer',
        data_root='/mnt/Datasets/UnrealStereo4K',
        split='splits/u4k/test.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))


test_out_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset_disp',
        mode='infer',
        data_root='/mnt/Datasets/UnrealStereo4K',
        split='splits/u4k/test_out.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))
