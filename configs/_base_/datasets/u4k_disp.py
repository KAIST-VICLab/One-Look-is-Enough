

train_dataloader=dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset_disp',
        mode='train',
        data_root='Datasets/UnrealStereo4K',
        split='splits/u4k/train.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            degree=1.0,
            random_crop=True, # random_crop_size will be set as patch_raw_shape
            network_process_size=[384, 512])))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset_disp',
        mode='infer',
        data_root='Datasets/UnrealStereo4K',
        split='splits/u4k/val.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))

val_consistency_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset_disp',
        mode='infer',
        data_root='Datasets/UnrealStereo4K',
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
        data_root='Datasets/UnrealStereo4K',
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
        data_root='Datasets/UnrealStereo4K',
        split='splits/u4k/test_out.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))
