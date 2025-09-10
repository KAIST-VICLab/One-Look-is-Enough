<div align="center">
<h2>[ICCV 2025] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images
</h2>

<div>    
    <a href='https://byeongjun1022.github.io/' target='_blank'>Byeongjun Kwon</a></sup>&nbsp&nbsp&nbsp&nbsp;
    <a href='https://scholar.google.com/citations?user=bGXte_4AAAAJ&hl=ko' target='_blank'>Munchurl Kim</a><sup>†</sup>
</div>
<br>
<div>
    <sup>†</sup>Corresponding author</span>
</div>
<div>
    <sup>1</sup>KAIST (Korea Advanced Institute of Science and Technology), South Korea</span>
</div>

<div>
    <h4 align="center">
        <a href="https://kaist-viclab.github.io/One-Look-is-Enough_site/" target='_blank'>
        <img src="https://img.shields.io/badge/🏠-Project%20Page-blue">
        </a>
        <a href="http://arxiv.org/abs/2503.22351" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2503.22351-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/One-Look-is-Enough">
    </h4>
</div>
</div>

---

<h4>
This repository is the official PyTorch implementation of "One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images". Our proposed method, PRO, achieves state-of-the-art zero-shot depth accuracy on high-resolution datasets with fine-grained details, outperforming existing depth refinement methods.
</h4>

---

## 📧 News
- **Sep 10, 2025:** Train code and Inference code are released
- **Jun 26, 2025:** "One Look is Enough" is accepted to ICCV 2025
- **Mar 28, 2025:** This repository is created

---

## Tested Environment
- OS: Ubuntu 20.04
- Python: 3.8
- PyTorch: 2.1.2 
- CUDA: 12.1 
- GPU: RTX 4090


## Environment setup
```bash
conda env create -n pro --file environment.yml
conda activate pro
```

## NOTE
Before running the code, please first run:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/One-Look-is-Enough"
export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/One-Look-is-Enough/external"
```
**Make sure that you have exported the `external` folder which stores codes from other repos (ZoeDepth, Depth-Anything V1, V2, etc.)**

## Pretrained Models

Pre-trained models need to be placed in the `./pretrained/` directory.

- **PRO.pth**: Trained on the **UnrealStereo4K** dataset. ([Download](https://www.dropbox.com/scl/fi/otwh4ne7s69zzb8s9qt32/PRO.pth?rlkey=gs9dbomwtgyoyets3moha83yq&st=3cmbmp6k&dl=0))
- **depth_anything_v2_vitl.pth**: Pre-trained **Depth-Anything-V2-Large** checkpoint. ([Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true))

### File Structure:
```bash
One-Look-is-Enough_private/
└── pretrained/
    ├── Depth-Anything-V2/
    │   └── depth_anything_v2_vitl.pth   # Depth-Anything V2 model
    └── PRO/
        └── PRO.pth                     # PRO model (trained on UnrealStereo4K)
```

## Running
To execute user inference, use the following command:

```bash
python tools/test_disp.py configs/test/test_general.py --cfg-option general_dataloader.dataset.rgb_image_dir='<img-directory>' [--save] [-save-residual] --work-dir <output-path> --test-type general --patch-split-num [h, w]
```
Arguments Explanation:
- `--cfg-option`: Specify the input image directory. Maintain the prefix as it indexes the configuration. (To learn more about this, please refer to [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html). Basically, we use MMEngine to organize the configurations of this repo).
- `--save`: Enable the saving of output files to the specified `--work-dir` directory (Make sure using it, otherwise there will be nothing saved).
- `--save-residual`: Enable the saving of residual outputs, but this option only works if `--save` is enabled. It saves the residual data alongside the regular output.
- `--work-dir`: Directory where the output files will be stored.
- `--patch-split-num`: Define how the input image is divided into smaller patches for processing. You can specify any patch size,`(h,w)`, where `h` is the height and `w` is the width. This helps control the granularity of image processing during inference. Default: `(4 4)`. 

## User Training

Please refer to [train](./docs/train.md) for more details.

## Results
Please visit our [project page](https://kaist-viclab.github.io/One-Look-is-Enough_site/) for more experimental results.

## Citation
If the content is useful, please cite our paper:
```bibtex
@misc{kwon2025onelook,
      title={One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images}, 
      author={Byeongjun Kwon and Munchurl Kim},
      year={2025},
      eprint={2503.22351},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.22351}, 
}
```

## License
The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr).

## Acknowledgement
This repository is built upon [FMA-Net](https://github.com/KAIST-VICLab/FMA-Net/), [C-DiffSET](https://github.com/KAIST-VICLab/C-DiffSET), and [PatchFusion](https://github.com/zhyever/PatchFusion).
We gratefully thank the [PatchFusion](https://github.com/zhyever/PatchFusion) authors for open-sourcing their code, which made our implementation and experiments much easier.