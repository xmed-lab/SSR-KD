## SSR-KD

Yiqun Lin, et al. "Real-Time, Population-Based Reconstruction of 3D Bone Models via Very-Low-Dose Protocols."

### 1. Code Structure

- `./data/*.py` are mainly used to preprocess CT, simulate X-ray projections, and generate segmentation masks from bone meshes. Processed data are saved in `./data/processed/`.
- `./bone_seg/` is used for X-ray-based bone segmentation.
- `./bone_recon/` is used for bone reconstruction.
- `./bone_recon/recon_ct/` is used for CT-based bone reconstruction.
- `./bone_recon/recon_xray` is used for 2-view X-ray-based bone reconstruction.

### 2. Environment Requirement

This repo was developed and tested on the following environment:

- **OS:** Ubuntu 24.04
- **GPU:** 8x NVIDIA GeForce 3090 (at least two 3090 GPUs are required for model training)
- **CUDA:** 11.1
- **PyTorch:** 1.8 (also tested with 1.13)

Additionally,  [TIGRE](https://github.com/CERN/TIGRE) is required for projection simulation, and [PyTorch3D](https://github.com/facebookresearch/pytorch3d) is required for calculating the chamfer distance (equal to ASSD).

### 3. License

This repository is released under MIT License (see LICENSE file for details).