# User Training

We provide training illustrations on Unreal4kStereo dataset in this document. Users can adopt to custome datasets based on this Unreal4kStereo version. We provide all configs.


## Dataset Preparation

Download the dataset from https://github.com/fabiotosi92/SMD-Nets.

Preprocess the dataset following the [instruction](https://github.com/fabiotosi92/SMD-Nets?tab=readme-ov-file#unrealstereo4k) (convert images to `raw` format).

```none
One-Look-is-Enough
├── estimator
├── docs
├── ...
├── Datasets
│   ├── UnrealStereo4K
│   │   ├── 00000
│   │   │   ├── Disp0
│   │   │   │   ├── 00000.npy
│   │   │   │   ├── 00001.npy
│   │   │   │   ├── ...
│   │   │   ├── Extrinsics0
│   │   │   ├── Extrinsics1
│   │   │   ├── Image0
│   │   │   │   ├── 00000.raw (Note it's important to convert png to raw to speed up training)
│   │   │   │   ├── 00001.raw
│   │   │   │   ├── ...
│   │   ├── 00001
│   │   │   ├── Disp0
│   │   │   ├── Extrinsics0
│   │   │   ├── Extrinsics1
│   │   │   ├── Image0
|   |   ├── ...
|   |   ├── 00008
```

## Prerequisites (for BFM Mask)
For BFM mask generation, you must precompute $E_c$, $E_{gt}$, and $M_{unreliable}$ used in Figure 4(c) of the paper.
Run the script below.

```bash
bash scripts/preprocess.sh
```

## Model Training

```bash
bash scripts/train.sh configs/PRO/train.py --work-dir ./output --log-name pro --tag 'pro'
```

- `--log-name`: experiment name shown in wandb website
- `--work-dir`: `work-dir + log-name` indicates the path to save logs and checkpoints
- `--tag`: tags shown in wandb website
- `--debug`: if set, omit wandb log