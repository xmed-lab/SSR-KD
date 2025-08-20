## SSR-KD

Yiqun Lin, et al. "Real-Time, Population-Based Reconstruction of 3D Bone Models via Very-Low-Dose Protocols." [arxiv](https://arxiv.org/abs/2508.13947)

```
@misc{lin2025realtimepopulationbasedreconstruction3d,
      title={Real-Time, Population-Based Reconstruction of 3D Bone Models via Very-Low-Dose Protocols}, 
      author={Yiqun Lin and Haoran Sun and Yongqing Li and Rabia Aslam and Lung Fung Tse and Tiange Cheng and Chun Sing Chui and Wing Fung Yau and Victorine R. Le Meur and Meruyert Amangeldy and Kiho Cho and Yinyu Ye and James Zou and Wei Zhao and Xiaomeng Li},
      year={2025},
      eprint={2508.13947},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2508.13947}, 
}
```

### 1. Code Structure

- `./data/*.py` are mainly used to preprocess CT, simulate X-ray projections, and generate segmentation masks from bone meshes. Processed data are saved in `./data/processed/`.
- `./bone_seg/` is used for X-ray-based bone segmentation.
- `./bone_recon/` is used for bone reconstruction.
- `./bone_recon/recon_ct/` is used for CT-based bone reconstruction.
- `./bone_recon/recon_xray` is used for 2-view X-ray-based bone reconstruction.

### 2. Data Structure

For labeled data:

````shell
├── ./data/processed/<case_id>/
│   ├── submesh/
│   │   ├── sub_0.obj # Patella
│   │   ├── sub_1.obj # Femur
│   │   ├── sub_2.obj # Fibula
│   │   └── sub_3.obj # Tibia
│   ├── mesh.obj      # merge bones
│   ├── projs.npz     # X-rays (512×521)
│   ├── bone_mask.npz # X-rays's bone masks (512×512)
│   ├── ct_256x.npz   # CT (256×256×256) used to train CT-based recon
│   ├── ct_512x.npz   # CT (512×512×512) used to generate X-rays
│   ├── seg_mask.npz  # CT's bone masks (512×512×512) used to generate X-rays' bone masks
│   ├── sampling.npz  # sampled points
│   └── spacing.npz   # spacing for CT
````

For unlabeled data:

````shell
├── ./data/processed/<case_id>/
│   ├── projs.npz   # X-rays (512×521)
│   ├── ct_256x.npz # CT (256×256×256) used to train CT-based recon
│   ├── ct_512x.npz # CT (512×512×512) used to generate X-rays
│   └── spacing.npz # spacing for mesh/CT
````

### 2. Environment Requirement

This repo was developed and tested on the following environment:

- **OS:** Ubuntu 24.04
- **GPU:** 8x NVIDIA GeForce 3090 (at least two 3090 GPUs are required for model training)
- **CUDA:** 11.1
- **PyTorch:** 1.8 (also tested with 1.13)

Additionally,  [TIGRE](https://github.com/CERN/TIGRE) is required for projection simulation, and [PyTorch3D](https://github.com/facebookresearch/pytorch3d) is required for calculating the chamfer distance (equal to ASSD).

### 3. Results

| Metric      | Average | Patella | Femur | Fibula | Tibia |
| ----------- | ------- | ------- | ----- | ------ | ----- |
| DSC (%) ↑   | 90.9    | 87.5    | 96.1  | 84.1   | 96.0  |
| HD (mm) ↓   | 2.76    | 3.30    | 2.20  | 2.87   | 2.06  |
| ASSD (mm) ↓ | 0.94    | 1.12    | 0.80  | 1.05   | 0.77  |

### 4. Model Weights

| Model      | Weight Link                                                  |
| ---------- | ------------------------------------------------------------ |
| bone_seg   | [237 MB](https://drive.google.com/file/d/1B_Jd1wqTN4cXMBTWbCJlzQTOkIdxyBRM/view?usp=sharing) |
| recon_ct   | [76 MB](https://drive.google.com/file/d/1MCO0lqHPTLCzkkxy16159lvbHyO1QHK2/view?usp=sharing) |
| recon_xray | [112 MB](https://drive.google.com/file/d/1_OySQkscbht-kj78JnaZMPRFQzPVmbHD/view?usp=sharing) |

### License

This repository is released under MIT License (see LICENSE file for details).
