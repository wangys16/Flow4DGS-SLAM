<p align="center">
  <h1 align="center">Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM</h1>
  <p align="center">
    <a href="https://wangys16.github.io">Yunsong Wang</a></span> ·
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a><sup></sup> <br>
    National University of Singapore<br>
  </p>
  <h2 align="center">CVPR 2026 Highlight</h2>
  <h3 align="center"><a href="https://github.com/wangys16/Flow4DGS-SLAM">Code</a> | <a href="https://arxiv.org/pdf/2604.22339">Paper</a> | <a href="https://wangys16.github.io/Flow4DGS-SLAM/">Project Page</a> </h3>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  </div>
</p>

<p align="center">
  <a href="">
    <img src="https://github.com/wangys16/Flow4DGS-SLAM/blob/main/assets/teaser_cropped.png" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
<strong>Efficient Dynamic Reconstruction + SLAM</strong>: Our Flow4DGS-SLAM solves the complex dynamic mapping and tracking task in a much more efficient pipeline.
</p>

## News:

- [2026/04/09] Flow4DGS-SLAM is selected as a <strong>Highlight</strong> paper 🚀.
- [2026/02/21] Flow4DGS-SLAM is accepted to <strong>CVPR 2026</strong> 🔥.


# 1.Installation
 
 
```
git clone https://github.com/wangys16/Flow4DGS-SLAM.git
cd Flow4DGS-SLAM
```

Setup the environment.
```
conda create -n dyslam python=3.8
conda activate dyslam
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

The simple-knn and diff-gaussian-rasterization libraries use the ones provided by MonoGS.
```
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

Use torch-batch-svd speed up (Optional)
```
git clone https://github.com/KinglittleQ/torch-batch-svd
cd torch-batch-svd
python setup.py install
cd ..
```

# 2.Pretrained Models

Download **YOLOv9e-seg**
```bash
cd pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt
```
Or download it directly from https://docs.ultralytics.com/models/yolov9/

Download **RAFT**

The model **raft-things.pth** used in this system can be obtained directly from https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT


# 3.Datasets

### TUM-RGBD dataset

Download the sequence using the following command:

```bash
bash scripts/download_tum.sh
```

### BONN dataset

Download the sequence using the following command:

```bash
bash scripts/download_bonn.sh
```

# 4.Testing

### TUM-RGBD dataset
```bash
bash scripts/train_tum.sh
```
### BONN dataset
```bash
bash scripts/train_bonn.sh
```



# 5.Acknowledgement
This work incorporates many open-source codes. We extend our gratitude to the authors of the software.
- [4DGS-SLAM](https://github.com/yanyan-li/4DGS-SLAM)
- [MonoGS](https://github.com/muskie82/MonoGS)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [GeoGaussian](https://github.com/yanyan-li/GeoGaussian)
- [SC-GS](https://github.com/CVMI-Lab/SC-GS)
- [Open3D](https://github.com/isl-org/Open3D)





# 6.Citation
If you find this code/work useful for your own research, please consider citing:
```
@article{wang2026flow4dgs,
  title={Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM},
  author={Wang, Yunsong and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2604.22339},
  year={2026}
}
```