# VENet
![VENet](https://github.com/zsc15/VENet/blob/main/figures/VENet.jpg)
![Early Gastric Cancer Diagnosis](https://github.com/zsc15/VENet/blob/main/figures/VENet_SVM_DL2.png)
## Experimental environment and dependencies
The VENet model was implemented under the open-source deep learning framework **PyTorch**. All models were trained and evaluated on a single **Nvidia Tesla A100 GPU**, **Ubuntu 18.04.6 LTS**.
- python 3.6.9
- numpy                     1.17.3
- scipy                     1.3.1
- imageio                   2.9.0
- joblib                    0.14.0
- opencv-python             4.4.0.46
- torch                     1.7.1+cu110
- scikit-image              0.17.2
- scikit-learn              0.20.4
- openslide-python          1.1.2
## Public datasets (Glas and CRAG)
1. Train the VENet model with Lv loss function on two public datasets
`python cg_train_VENet_Lv_adam.py --exp VENet_sigmoid`
2. Test the VENet  model
`python inference_dataset.py --exp VENet_sigmoid --test_root_path image_path --test_mask_path mask_path`
3. Visualize one image or compared with other different models, run this notebook `visualize.ipynb`
## Nanfang Hospital's datasets
69 test WSIs are available at  [IEEE DataPort](https://dx.doi.org/10.21227/ngxk-6952) or [Mendeley data](http://dx.doi.org/10.17632/y8gk8dmf7y.1). 30000 pathological images with annotations are used for training VENet. These training images with annotations are available at [IEEE DataPort](https://dx.doi.org/10.21227/rkqj-zd61).
1 or [Mendeley data](http://dx.doi.org/10.17632/mnjxs334pv.1). We use `train_VENet_nanfang_datasets.py` to train VENet on 30000 self-collected pathological images with annotations, then obtain the pretrained weights named **best.pth**([Google Cloud](https://drive.google.com/file/d/178SvJQb6BiV8_x6FrD6qHi66_16xI0pA/view?usp=share_link)).
2. Load the pretrained weights. Please directly run this critical python code: `fastread2.py`. Generate six cache files: three **npy** files about pathological feature, three **.xml**  suffix files for annotation , nucleus width label, and final detection results of whole slide image.
3. Use **Aperio ImageScope** software for visualization.
## Citation
If you find our code is useful for you, please cite us.
```
@article{ZHANG2024108178,
title = {VENet: Variational energy network for gland segmentation of pathological images and early gastric cancer diagnosis of whole slide images},
journal = {Computer Methods and Programs in Biomedicine},
volume = {250},
pages = {108178},
year = {2024},
issn = {0169-2607},
doi = {https://doi.org/10.1016/j.cmpb.2024.108178},
url = {https://www.sciencedirect.com/science/article/pii/S0169260724001743},
author = {Shuchang Zhang and Ziyang Yuan and Xianchen Zhou and Hongxia Wang and Bo Chen and Yadong Wang},
keywords = {Gland segmentation, Early gastric cancer diagnosis, Whole slide images, VENet}
}

```
