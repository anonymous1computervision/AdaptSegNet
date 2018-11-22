# Domain Adapt for Semantic Segmentation

Pytorch implementation domain adaption of semantic segmentation from the synthetic dataset (source domain) to the real dataset (target domain).

Contact: Yong-Xiang Lin (xiaosean5408 at gmail dot com)

<!-- [Learning to Adapt Structured Output Space for Semantic Segmentation](https://arxiv.org/abs/1802.10349) <br />
[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home)\*, [Wei-Chih Hung](https://hfslyc.github.io/)\*, [Samuel Schulter](https://samschulter.github.io/), [Kihyuk Sohn](https://sites.google.com/site/kihyuksml/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) and [Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/) <br />
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 (**spotlight**) (\* indicates equal contribution). -->

## Paper

<!-- Please cite our paper if you find it useful for your research.

```
@article{Tsai_adaptseg_2018,
  author = {Y.-H. Tsai and W.-C. Hung and S. Schulter and K. Sohn and M.-H. Yang and M. Chandraker},
  journal = {arXiv preprint arXiv:1802.10349},
  title = {Learning to Adapt Structured Output Space for Semantic Segmentation},
  year = {2018}
}
``` -->

## Prerequisites
- Python 3
- NVIDIA GPU (10G up, I use Nvidia GTX 1080Ti) + CUDA cuDNN
- Pytorch(Vesion needs 0.4.1 up because I use spectral_norm)

## Getting Started
### Installation

#### You can choose [pip install / conda install by yml]()
* #### Intall method 1. use pip 
    - Install PyTorch and dependencies from http://pytorch.org
    ```bash
    pip install dominate, scipy, matplotlib, pillow, pyyaml
    ```
* ####  Intall method 2. use conda env 
    ```bash
    conda env create -f environment.yml
    ```
    - On Windows
        ```bash
        activate Gated-AdaptSeg 
        ```
    - On Linux
        ```bash
        conda activate Gated-AdaptSeg 
        ```
    ```bash
    pip install dominate
    ```

- Clone this repo:
```bash
git clone https://github.com/xiaosean/Gated-AdaptSeg
cd Gated-AdaptSeg
```

## Dataset
* Download the [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) as the source domain, and put it in the `data/GTA5` folder

* Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) as the target domain, and put it in the `data/Cityscapes` folder

## Testing
* Download the pre-trained (comming soon) and put it in the `model` folder

* Test the model and results will be saved in the `snapshot` folder, and remember replace xxxx.pth as yours(e.g. ./snapshot/GTA2Cityscapes_multi-GTA5_250000.pth)

```
python evaluate_cityscapes.py --restore-from ./snapshot/GTA2Cityscapes_multi-GTA5_XXXX.pth
```

* Compute the IoU on Cityscapes (thanks to the code from [VisDA Challenge](http://ai.bu.edu/visda-2017/))
```
python compute_iou.py ./data/Cityscapes/data/gtFine/val result/cityscapes
```

## Training Examples
* Train the GTA5-to-Cityscapes model - 
    * Setting default use => configs/default_edge_TTUR.yaml
```
python train.py
```
## Related Implementation and Dataset


* Y.-H. Chen, W.-Y. Chen, Y.-T. Chen, S. Schulter, K. Sohn, and M.-H. Yang. NLearning to Adapt Structured Output Space for Semantic Segmentation. In CVPR 2018. [[paper]](https://arxiv.org/abs/1802.10349) [[code]](https://github.com/wasidennis/AdaptSegNet) 

* W.-C. Hung, Y.-H Tsai, Y.-T. Liou, Y.-Y. Lin, and M.-H. Yang. Adversarial Learning for Semi-supervised Semantic Segmentation. In ArXiv, 2018. [[paper]](https://arxiv.org/abs/1802.07934) [[code]](https://github.com/hfslyc/AdvSemiSeg)
* Y.-H. Chen, W.-Y. Chen, Y.-T. Chen, B.-C. Tsai, Y.-C. Frank Wang, and M. Sun. No More Discrimination: Cross City Adaptation of Road Scene Segmenters. In ICCV 2017. [[paper]](https://arxiv.org/abs/1704.08509) [[project]](https://yihsinchen.github.io/segmentation_adaptation/)

## Acknowledgment
This code is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).
Visualize part is heavily borrowed from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).

## Note
The model and code are available for non-commercial research purposes only.
* 11/22/2018: Add readme.txt




