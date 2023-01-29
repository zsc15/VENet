# VENet
The source code  for VENet. Details will be refined soon.
## Public dataset (Glas and CRAG)
1. train the VENet model with Lv loss function on two public datasets
`python cg_train_VENet_Lv_adam.py --exp VENet_sigmoid`
2. test the VENet  model
`python inference_dataset.py --exp VENet_sigmoid --test_root_path image_path --test_mask_path mask_path`
3. visualize one image or compared with other different models, run this notebook `visualize.ipynb`
## Nanfang Hospital
Please directly run this critical python code: `fastread2.py`. Generate six cache files: 3 'npy' files about pathological feature, 3 '.xml'  suffix files for annotation , nucleus width label, and final detection results of Whole Slide Image.
Use **Aperio ImageScope** software for visualization.
## Citation
If you find our code is useful for you, please cite us.
