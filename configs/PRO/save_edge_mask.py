_base_ = [
    '../_base_/datasets/u4k_disp.py',
]

#^ u4k val test할 때에
val_dataloader=dict(
    dataset=dict(
        split='splits/u4k/train.txt',
        # mode='train',
        resize_mode='depth-anything',
        transform_cfg=dict(
            network_process_size=[518, 518])))



