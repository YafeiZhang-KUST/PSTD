
#Single Image Super-Resolution Reconstruction with Preservation of Structure and Texture Details

This package contains the source code which is associated with the following paper:

Edited by YuQing Huang

Usage of this code is free for research purposes only. 

Thank you.

# Requirements:
    CUDA  11.4
    Python  3.7
    Pytorch  1.7.0
    torchvision  0.8.2
    numpy  1.16.2

# Get Started
## 1.Install:
    download the code
    git clone https://github.com/YafeiZhang-KUST/CMReID.git
    cd PSTD
    
## 2.Datasets
- SYSU-MM01
- RegDB
## 3.Results
Dataset | Rank1  | mAP | mINP
 ---- | ----- | ------  | -----
 RegDB | ~95.12% | ~91.06%  | ~83.71%
 SYSU-MM01  | ~61.67% | ~57.72% | ~42.30%
## 4.Training
Train a model by
```bash
python train.py 
You can download the weight files that have been trained(链接：https://pan.baidu.com/s/1JMkIhBMZQdVOjqI_6fmycQ 
password：2022)

```
# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Linbo Shi
    Faculty of Information Engineering and Automation
    Kunming University of Science and Technology                                                           
    Email: 1527467911@qq.com

# Acknowledgements
Our code is based on https://github.com/mangye16/Cross-Modal-Re-ID-baseline.

