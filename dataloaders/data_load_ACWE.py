import os
import torch
import numpy as np
import scipy
import torchvision
import cv2
from dataloaders.augmentations import ElasticTransformations, RandomRotationWithMask
from torch.utils.data import Dataset
from PIL import Image
from cd_Utils import Preprocess as pre
import utils.util as util
import itertools
import staintools
import math
import random
from torch.utils.data.sampler import Sampler
cv2.setNumThreads(0)


class Gland_ACWE_seg(Dataset):
    """ LA Dataset """
    def __init__(self, args, base_dir=None, split='train', num=None):
        self.args = args
        self._base_dir = base_dir
        self.sample_list = []
        self.H_size = 400
        self.image_list = os.listdir('../../datasets/Glas/train/images/')
        self.scales = self.args.scale
        self.count = 0
        self.bs = self.args.batch_size
        # self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        image = Image.open(self._base_dir+'/Glas/train/images/'+image_name)
        label = Image.open(self._base_dir+'/Glas/train/anno/'+image_name)
        if self.count % self.bs == 0:
            # sf = random.choice([1,.9,.8,.7,.6])
            self.sf = random.choice(self.scales)
            # self.count = 0  # optional
            self.patch_size = int(self.H_size*self.sf)
        self.count += 1
        image = np.array(image)
        # image = image/255
        H, W, _ = image.shape
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        # print(rnd_h, rnd_w, self.patch_size)
        ###### randomly crop the patch
        patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        label = np.array(label)
        # print(image.shape, label.shape)
        patch_L = label[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        ######
        ###### augmentation - flip, rotate for patch and its annotation
        mode = np.random.randint(0, 8)
        patch_H = util.augment_img(patch_H, mode=mode)
        patch_L = util.augment_img(patch_L, mode=mode)
        ######
        patch_L[patch_L > 0] = 1
        patch_H = np.float32(patch_H / 255.)
        # patch_H = patch_H.transpose((2, 0, 1))
        patch_H = torch.from_numpy(np.ascontiguousarray(patch_H)).permute(2, 0, 1).float()
        patch_L = torch.from_numpy(np.ascontiguousarray(patch_L)).long().unsqueeze(0)
        # patch_L = label.unsqueeze(dim=0)
        sample = {'image': patch_H, 'label': patch_L}
        return sample

class Gland_ACWE_seg_cg(Dataset):##GlaS dataset and CRAG dataset
    """ LA Dataset """
    def __init__(self, args, base_dir=None, local_rank=0, split='train', num=None):
        self.local_rank = local_rank
        self.args = args
        self._base_dir = base_dir
        self.sample_list = []
        # self.patch_size
        self.H_size = 600
        # self.image_list = os.listdir('../../datasets/Glas/train/images/')
        ##############################obtain overall images from GlaS dataset and CRAG dataset
        self.GlaS_root_img_path = self._base_dir + '/Glas/train/images/'
        self.GlaS_root_anno_path = self._base_dir + '/Glas/train/anno/'
        self.CRAG_root_img_path = self._base_dir + '/CRAG/train/images/'
        self.CRAG_root_anno_path = self._base_dir + '/CRAG/train/anno/'
        self.GlaS_images_list = os.listdir('../../datasets/Glas/train/images/')
        self.CRAG_images_list = os.listdir('../../datasets/CRAG/train/images/')
        self.original_imglist = [os.path.join(self.GlaS_root_img_path, name) for name in self.GlaS_images_list] + \
                                [os.path.join(self.CRAG_root_img_path, name) for name in self.CRAG_images_list]
        self.original_annolist = [os.path.join(self.GlaS_root_anno_path, name) for name in self.GlaS_images_list] + \
                                 [os.path.join(self.CRAG_root_anno_path, name) for name in self.CRAG_images_list]
        self.num = len(self.original_imglist)
        ################################
        self.scales = self.args.scale
        self.count = 0
        self.bs = self.args.batch_size
        # self.permutation = np.load('permutation_'+str(args.N)+'.npy') ###fixed
        # self.image_list = [self.original_imglist[i] for i in self.permutation]
        # self.anno_list = [self.original_annolist[i] for i in self.permutation]
        self.image_list = self.original_imglist
        self.anno_list = self.original_annolist
        if self.num is not None:
            self.image_list = self.image_list[:self.num]
            self.anno_list = self.anno_list[:self.num]#####偶数
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        anno_name = self.anno_list[idx]
        image = Image.open(image_name)
        label = Image.open(anno_name)
        image = image.resize((self.H_size, self.H_size), Image.ANTIALIAS)
        label = label.resize((self.H_size, self.H_size), Image.ANTIALIAS)
        if self.count % self.bs == 0:
            # sf = random.choice([1,.9,.8,.7,.6])
            self.sf = random.choice(self.scales)
            # self.count = 0  # optional
            self.patch_size = int(self.H_size*self.sf)
        self.count += 1
        image = np.array(image)
        # image = image/255
        W, H, _ = image.shape
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        # print(rnd_h, rnd_w, self.patch_size)
        ###### randomly crop the patch
        patch_H = image[rnd_w:rnd_w + self.patch_size, rnd_h:rnd_h + self.patch_size, :]
        label = np.array(label)
        # print(image.shape, label.shape)
        patch_L = label[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        ######
        ###### augmentation - flip, rotate for patch and its annotation
        mode = np.random.randint(0, 8)
        patch_H = util.augment_img(patch_H, mode=mode)
        patch_L = util.augment_img(patch_L, mode=mode)
        ######
        patch_L[patch_L > 0] = 1
        patch_H = np.float32(patch_H / 255.)
        # patch_H = patch_H.transpose((2, 0, 1))
        patch_H = torch.from_numpy(np.ascontiguousarray(patch_H)).permute(2, 0, 1).float()
        patch_L = torch.from_numpy(np.ascontiguousarray(patch_L)).long().unsqueeze(0)
        # patch_L = label.unsqueeze(dim=0)
        sample = {'image': patch_H, 'label': patch_L}
        return sample

class Gland_ACWE_seg_cg_loader(Dataset):##GlaS dataset and CRAG dataset
    """ LA Dataset """
    def __init__(self, args, base_dir=None, local_rank=0, validation=False):
        self.local_rank = local_rank
        self.validation = validation
        self.args = args
        self._base_dir = base_dir
        self.sample_list = []
        self.H_size = 600
        # self.image_list = os.listdir('../../datasets/Glas/train/images/')
        ##############################obtain overall images from GlaS dataset and CRAG dataset
        self.GlaS_root_img_path = self._base_dir + '/Glas/train/images/'
        self.GlaS_root_anno_path = self._base_dir + '/Glas/train/anno/'
        self.CRAG_root_img_path = self._base_dir + '/CRAG/train/images/'
        self.CRAG_root_anno_path = self._base_dir + '/CRAG/train/anno/'
        self.GlaS_images_list = os.listdir('../../datasets/Glas/train/images/')
        self.CRAG_images_list = os.listdir('../../datasets/CRAG/train/images/')
        self.original_imglist = [os.path.join(self.GlaS_root_img_path, name) for name in self.GlaS_images_list] + \
                                [os.path.join(self.CRAG_root_img_path, name) for name in self.CRAG_images_list]
        self.original_annolist = [os.path.join(self.GlaS_root_anno_path, name) for name in self.GlaS_images_list] + \
                                 [os.path.join(self.CRAG_root_anno_path, name) for name in self.CRAG_images_list]
        self.num = len(self.original_imglist)
        ################################
        self.scales = self.args.scale
        self.count = 0
        self.bs = self.args.batch_size
        self.permutation = np.load('permutation_'+str(args.N)+'.npy') ###fixed
        self.image_list = [self.original_imglist[i] for i in self.permutation]
        self.anno_list = [self.original_annolist[i] for i in self.permutation]
        self.image_mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            RandomRotationWithMask(45, resample=False, expand=False, center=None),
            ElasticTransformations(2000, 60),
            torchvision.transforms.ToTensor()
        ])
        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor()
        ])

        if self.num is not None:
            self.image_list = self.image_list[:self.num]
            self.anno_list = self.anno_list[:self.num]#####偶数
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        anno_name = self.anno_list[idx]
        image = Image.open(image_name)
        label = Image.open(anno_name)
        # image = image.resize((self.H_size, self.H_size), Image.ANTIALIAS)
        # label = label.resize((self.H_size, self.H_size), Image.ANTIALIAS)
        image = np.array(image)
        label = np.array(label)
        image, mask = self.pad_image(image, label)  #####  600 X 600 resolution
        # print(image.shape)
        if self.count % self.bs == 0:
            # sf = random.choice([1,.9,.8,.7,.6])
            self.sf = random.choice(self.scales)
            # self.count = 0  # optional
            self.patch_size = int(self.H_size*self.sf)
        self.count += 1
        # image = image/255
        H, W, _ = image.shape
        # print(H, W, self.patch_size)
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        # print(rnd_h, rnd_w, self.patch_size)
        ###### randomly crop the patch
        patch_H = image[rnd_w:rnd_w + self.patch_size, rnd_h:rnd_h + self.patch_size, :]
        # print(image.shape, label.shape)
        patch_L = mask[rnd_w:rnd_w + self.patch_size, rnd_h:rnd_h + self.patch_size]
        ######
        # print(patch_L.shape)
        patch_L, patch_H = self.apply_data_augmentation(patch_H, patch_L)
        # print(patch_H.shape, patch_L.shape)
        label_L = self.create_eroded_mask(patch_L)
        ###### augmentation - flip, rotate for patch and its annotation
        # mode = np.random.randint(0, 8)
        # patch_H = util.augment_img(patch_H, mode=mode)
        # patch_L = util.augment_img(patch_L, mode=mode)
        # ######
        # patch_L[patch_L > 0] = 1
        # patch_H = np.float32(patch_H / 255.)
        # # patch_H = patch_H.transpose((2, 0, 1))
        # patch_H = torch.from_numpy(np.ascontiguousarray(patch_H)).permute(2, 0, 1).float()
        # patch_L = torch.from_numpy(np.ascontiguousarray(patch_L)).long().unsqueeze(0)
        # patch_L = label.unsqueeze(dim=0)
        sample = {'image': patch_H, 'label': label_L}
        return sample


    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

            # we pad more than needed to later do translation augmentation
        pad_h = max((self.H_size - image.shape[0]) // 2, 0)
        pad_w = max((self.H_size - image.shape[1]) // 2, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def apply_data_augmentation(self, image, mask):
        """Helper function to apply all configured data augmentations on both mask and image"""
        patch = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
        n_glands = mask.max()
        label = torch.from_numpy(mask).float() / n_glands

        if not self.validation:
            patch_label_concat = torch.cat((patch, label[None, :, :].float()))
            patch_label_concat = self.image_mask_transforms(patch_label_concat)
            patch, label = patch_label_concat[0:3], np.round(patch_label_concat[3] * n_glands)
            patch = self.image_transforms(patch)
        else:
            label *= n_glands
        return label, patch

    def create_eroded_mask(self, label):
        """Helper function to create a mask where every gland is eroded"""
        boundaries = torch.zeros(label.shape)
        for i in np.unique(label):
            if i == 0: continue  # the first label is background
            gland_mask = (label == i).float()
            # print(gland_mask.shape)
            binarized_mask_border = scipy.ndimage.morphology.binary_erosion(gland_mask,
                                                                            structure=np.ones((13, 13)),
                                                                            border_value=1)

            binarized_mask_border = torch.from_numpy(binarized_mask_border.astype(np.float32))
            boundaries[label == i] = binarized_mask_border[label == i]

        label = (label > 0).float()
        label = torch.stack((boundaries, label))
        return label

class Gland_ACWE_original_img_loader(Dataset):##GlaS dataset and CRAG dataset
    """ LA Dataset """
    def __init__(self, args, base_dir=None, local_rank=0, validation=False):
        self.local_rank = local_rank
        self.validation = validation
        self.args = args
        self._base_dir = base_dir
        self.sample_list = []
        self.H_glas_size = 416
        self.H_crag_size = 512
        self.H_size = 600
        # self.image_list = os.listdir('../../datasets/Glas/train/images/')
        ##############################obtain overall images from GlaS dataset and CRAG dataset
        self.GlaS_root_img_path = self._base_dir + '/GlaS/'
        self.GlaS_root_anno_path = self._base_dir + '/GlaS/'
        self.CRAG_root_img_path = self._base_dir + '/CRAG1/train/Images/'
        self.CRAG_root_anno_path = self._base_dir + '/CRAG1/train/Annotation/'
        self.GlaS_images_list = ['train_'+str(i+1)+'.bmp' for i in range(85)]
        self.GlaS_annos_list = ['train_' + str(i + 1) + '_anno.bmp' for i in range(85)]
        self.CRAG_images_list = os.listdir('../../datasets/CRAG1/train/Images/')
        self.original_imglist = [os.path.join(self.GlaS_root_img_path, name) for name in self.GlaS_images_list] + \
                                [os.path.join(self.CRAG_root_img_path, name) for name in self.CRAG_images_list]
        self.original_annolist = [os.path.join(self.GlaS_root_anno_path, name) for name in self.GlaS_annos_list] + \
                                 [os.path.join(self.CRAG_root_anno_path, name) for name in self.CRAG_images_list]
        self.num = len(self.original_imglist)
        ################################
        self.scales = self.args.scale
        self.count = 0
        self.bs = self.args.batch_size
        # self.permutation = np.load('permutation_'+str(args.N)+'.npy') ###fixed
        # self.image_list = [self.original_imglist[i] for i in self.permutation]
        # self.anno_list = [self.original_annolist[i] for i in self.permutation]
        self.image_list = self.original_imglist
        self.anno_list = self.original_annolist
        self.image_mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomVerticalFlip(),
            RandomRotationWithMask(45, resample=False, expand=False, center=None),
            ElasticTransformations(2000, 60),
            torchvision.transforms.ToTensor()
        ])
        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor()
            # torchvision.transforms.Normalize([0.78780321, 0.5120167, 0.78493782], [0.16766301, 0.24838048, 0.13225162]),
            # torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
        ])

        if self.num is not None:
            self.image_list = self.image_list[:self.num]
            self.anno_list = self.anno_list[:self.num]#####偶数
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        anno_name = self.anno_list[idx]
        image = Image.open(image_name)
        label = Image.open(anno_name)
        # image = image.resize((self.H_size, self.H_size), Image.ANTIALIAS)
        # label = label.resize((self.H_size, self.H_size), Image.ANTIALIAS)
        image = np.array(image)
        mask = np.array(label)
        # image, mask = self.pad_image(image, label)  #####  600 X 600 resolution
        # print(image.shape)
        if self.count % self.bs == 0:
            # sf = random.choice([1,.9,.8,.7,.6])
            self.sf = random.choice(self.scales)
            # self.count = 0  # optional
            self.patch_size = math.ceil(self.sf * self.H_size)
            self.mode = np.random.randint(0, 8)
        self.count += 1
        # image = image/255
        H, W, _ = image.shape
        # print(W, H)
        if H < 800 and W < 800 and self.patch_size >= 600: #Glas dataset and 600pix
            image, mask = self.pad_image(image, mask)
            # print(image.shape)
            image = cv2.resize(image, (775, 600))
            mask = cv2.resize(mask, (775, 600))
            rnd_h = random.randint(0, max(0, 600 - self.patch_size))
            rnd_w = random.randint(0, max(0, 775 - self.patch_size))
            patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        else:## < 600pix and CRAG dataset, Glasdataset
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        # rnd_h = random.randint(0, max(0, H - self.patch_size))
        # rnd_w = random.randint(0, max(0, W - self.patch_size))
        # # print(rnd_h, rnd_w, self.patch_size)
        # ###### randomly crop the patch
        # patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        # # print(image.shape)
        # patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        ######
        ###### augmentation - flip, rotate for patch and its annotation
        patch_H = util.augment_img(patch_H, mode=self.mode)
        patch_L = util.augment_img(patch_L, mode=self.mode)
        # print(patch_H.shape)
        patch_L, patch_H = self.apply_data_augmentation(patch_H, patch_L)
        ######
        # print(patch_H.shape, patch_L.shape)
        label_L = self.create_eroded_mask(patch_L)
        ###### augmentation - flip, rotate for patch and its annotation
        # mode = np.random.randint(0, 8)
        # patch_H = util.augment_img(patch_H, mode=mode)
        # patch_L = util.augment_img(patch_L, mode=mode)
        # ######
        # patch_L[patch_L > 0] = 1
        # patch_H = np.float32(patch_H / 255.)
        # # patch_H = patch_H.transpose((2, 0, 1))
        # patch_H = torch.from_numpy(np.ascontiguousarray(patch_H)).permute(2, 0, 1).float()
        # patch_L = torch.from_numpy(np.ascontiguousarray(patch_L)).long().unsqueeze(0)
        # patch_L = label.unsqueeze(dim=0)
        sample = {'image': patch_H, 'label': label_L}
        return sample


    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

            # we pad more than needed to later do translation augmentation
        pad_h = max((self.H_size - image.shape[0]) // 2, 0)
        pad_w = max((self.H_size - image.shape[1]) // 2, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def apply_data_augmentation(self, image, mask):
        """Helper function to apply all configured data augmentations on both mask and image"""
        img = np.ascontiguousarray(image).transpose(2, 0, 1)
        patch = torch.from_numpy(img).float() / 255
        n_glands = mask.max()
        label = torch.from_numpy(np.ascontiguousarray(mask)).float() / n_glands if n_glands > 0 \
            else torch.from_numpy(np.ascontiguousarray(mask))
        # print(patch.shape, label.shape)
        if not self.validation:
            patch_label_concat = torch.cat((patch, label[None, :, :].float()))
            patch_label_concat = self.image_mask_transforms(patch_label_concat)
            patch, label = patch_label_concat[0:3], np.round(patch_label_concat[3] * n_glands)
            patch = self.image_transforms(patch)
        else:
            label *= n_glands
        return label, patch

    def create_eroded_mask(self, label):
        """Helper function to create a mask where every gland is eroded"""
        boundaries = torch.zeros(label.shape)
        for i in np.unique(label):
            if i == 0: continue  # the first label is background
            gland_mask = (label == i).float()
            # print(gland_mask.shape)
            binarized_mask_border = scipy.ndimage.morphology.binary_erosion(gland_mask,
                                                                            structure=np.ones((13, 13)),
                                                                            border_value=1)

            binarized_mask_border = torch.from_numpy(binarized_mask_border.astype(np.float32))
            boundaries[label == i] = binarized_mask_border[label == i]

        label = (label > 0).float()
        label = torch.stack((boundaries, label))
        return label



class Gland_seg2(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', patch_size=384, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.patch_size = patch_size

        self.image_list = os.listdir('../../datasets/Glas/train/images/')
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        # h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        # h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = Image.open(self._base_dir+'/Glas/train/images/'+image_name)
        label = Image.open(self._base_dir+'/Glas/train/anno/'+image_name)
        image = np.array(image)
        H, W, _ = image.size
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        image = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        image = image.transpose((2, 0, 1))
        image = image/255
        label = np.array(label)
        # print(image.shape, label.shape)
        label[label>0]=1
        image = torch.from_numpy(image).float().contiguous()
        label = torch.from_numpy(label).long().contiguous()
        label = label.unsqueeze(dim=0)
        sample = {'image': image, 'label': label}
        # if self.transform:
        #     sample = self.transform(sample)
        # print(sample['label'].shape)
        return sample

class Gland_ACWE_original_img_loader2(Dataset):##GlaS dataset and CRAG dataset
    """ LA Dataset """
    def __init__(self, args, base_dir=None, local_rank=0, validation=False, is_cn=True):
        self.is_cn = is_cn
        self.local_rank = local_rank
        self.validation = validation
        self.args = args
        self._base_dir = base_dir
        # self.sample_list = []
        self.H_glas_size = 416
        self.H_size = 600
        self.H_crag_size = 512
        # self.image_list = os.listdir('../../datasets/Glas/train/images/')
        ##############################obtain overall images from GlaS dataset and CRAG dataset
        self.GlaS_root_img_path = self._base_dir + '/GlaS/'
        self.GlaS_root_anno_path = self._base_dir + '/GlaS/'
        self.CRAG_root_img_path = self._base_dir + '/CRAG1/train/Images/'
        self.CRAG_root_anno_path = self._base_dir + '/CRAG1/train/Annotation/'
        self.GlaS_images_list = ['train_' + str(i + 1)+'.bmp' for i in range(85)]
        self.GlaS_annos_list = ['train_' + str(i + 1) + '_anno.bmp' for i in range(85)]
        self.CRAG_images_list = os.listdir('../../datasets/CRAG1/train/Images/')
        self.original_imglist = [os.path.join(self.GlaS_root_img_path, name) for name in self.GlaS_images_list] + \
                                [os.path.join(self.CRAG_root_img_path, name) for name in self.CRAG_images_list]
        self.original_annolist = [os.path.join(self.GlaS_root_anno_path, name) for name in self.GlaS_annos_list] + \
                                 [os.path.join(self.CRAG_root_anno_path, name) for name in self.CRAG_images_list]
        self.num = len(self.original_imglist)
        ################################
        self.scales = self.args.scale
        self.count = 0
        self.bs = self.args.batch_size
        # self.permutation = np.load('permutation_'+str(args.N)+'.npy') ###fixed
        # self.image_list = [self.original_imglist[i] for i in self.permutation]
        # self.anno_list = [self.original_annolist[i] for i in self.permutation]
        self.image_list = self.original_imglist
        self.anno_list = self.original_annolist
        self.image_mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            RandomRotationWithMask(45, resample=False, expand=False, center=None),
            ElasticTransformations(2000, 60),
            torchvision.transforms.ToTensor()
        ])
        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor()
        ])

        if self.num is not None:
            self.image_list = self.image_list[:self.num]
            self.anno_list = self.anno_list[:self.num]#####偶数
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        anno_name = self.anno_list[idx]
        image = Image.open(image_name)
        label = Image.open(anno_name)
        image = np.array(image)
        label = np.array(label)
        target = Image.open(self._base_dir + '/GlaS/'+'testA_35.bmp')
        if self.count % self.bs == 0:
            # sf = random.choice([1,.9,.8,.7,.6])
            self.sf = random.choice(self.scales)
            # self.count = 0  # optional
            self.patch_size = math.ceil(self.sf * self.H_size)
        self.count += 1
        # target = target.resize((self.patch_size, self.patch_size), Image.ANTIALIAS)
        # target = np.array(target)

        # image = image/255
        H, W, _ = image.shape
        # print(H, W, self.patch_size)
        ######
        if H < 800 and W < 800 and self.patch_size >= 600: #Glas dataset and 600pix
            image, label = self.pad_image(image, label)
            # print(image.shape)
            image = cv2.resize(image, (800, 600))
            label = cv2.resize(label, (800, 600))
            rnd_h = random.randint(0, max(0, 600 - self.patch_size))
            rnd_w = random.randint(0, max(0, 800 - self.patch_size))
            patchH = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patchL = label[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        else:## < 600pix and CRAG dataset, Glasdataset
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patchH = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patchL = label[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        ###### augmentation - flip, rotate for patch and its annotation
        mode = np.random.randint(0, 8)
        patch_H = util.augment_img(patchH, mode=mode)
        patch_L = util.augment_img(patchL, mode=mode)
        # print(patch_H.shape)
        if self.is_cn:
            # patch_H = self.color_normalization(target, patch_H)
            norm_patch_H = self.prepocess(patch_H)
            norm_patch_H = self.normalize_img(norm_patch_H)
        patch_L[patch_L>0]=1
        # print(np.max(patch_L))
        contour_mask = self.obtatin_boundray(patch_L)
        contour_mask = self.dilation(contour_mask, radius=3)
        patch_H = np.float32(patch_H / 255.)
        patch_H = torch.from_numpy(np.ascontiguousarray(patch_H)).permute(2, 0, 1).float()
        patch_L = torch.from_numpy(np.ascontiguousarray(patch_L)).long().unsqueeze(0)
        contour_mask = torch.from_numpy(np.ascontiguousarray(contour_mask)).long().unsqueeze(0)
        norm_patch_H = torch.from_numpy(np.ascontiguousarray(norm_patch_H)).permute((2, 0, 1)).float()
        # patch_L, patch_H = self.apply_data_augmentation(patch_H, patch_L)
        # patch = torch.from_numpy(patch_H).float() / 255
        # n_glands = patch_L.max()
        # label = torch.from_numpy(patch_L).float()
        ######
        # patch_L = patch_L.unsqueeze(dim=0)
        sample = {'image': patch_H, 'norm': norm_patch_H, 'label': patch_L, 'contour': contour_mask}
        return sample

    def dilation(self, x, radius=3):
        """ Return greyscale morphological dilation of an image,
        see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`_.

        Parameters
        -----------
        x : 2D array image.
        radius : int for the radius of mask.
        """
        from skimage.morphology import disk, dilation
        mask = disk(radius)
        x = dilation(x, footprint=mask)
        return x

    def obtatin_boundray(self, gt_mask):
        contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros_like(gt_mask)
        mask2 = np.zeros_like(np.ascontiguousarray(gt_mask))
        cv2.drawContours(mask2, contours, -1, (255, 0, 0), 1)
        # n = len(contours)
        # c = contours[0]
        # c = np.squeeze(c)
        # for i in range(n-1):
        #     c_i = contours[i+1]
        #     c_i = np.squeeze(c_i)
        #     c = np.concatenate((c, c_i), axis=0)
        # x = c[:, 0]
        # y = c[:, 1]
        # mask[y, x] = 255
        return mask2/255.0

    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

            # we pad more than needed to later do translation augmentation
        pad_h = max((600 - image.shape[0]) // 2, 0)
        pad_w = max((800 - image.shape[1]) // 2, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def normalize_img(self, img):
        if np.max(img) != np.min(img):
            trans_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            trans_img = img
        return trans_img

    def prepocess(self, img):
        return pre(img)


class Gland_ACWE_original_img_npy_loader(Dataset):##GlaS dataset and CRAG dataset
    """ LA Dataset """
    def __init__(self, args, base_dir=None, local_rank=0, validation=False, is_cn=True):
        self.is_cn = is_cn
        self.local_rank = local_rank
        self.validation = validation
        self.args = args
        self._base_dir = base_dir
        # self.sample_list = []
        # self.H_glas_size = 416
        self.H_size = 600
        # self.H_crag_size = 512
        # self.image_list = os.listdir('../../datasets/Glas/train/images/')
        ##############################obtain overall images from GlaS dataset and CRAG dataset
        self.img_npy_path = self._base_dir + '/GlaS_CRAG_cn_npy/train/images/'
        self.img_name_list = os.listdir(self.img_npy_path)
        self.original_imglist = [os.path.join(self._base_dir+'/GlaS_CRAG_cn_npy/train/', 'images', name) for name in self.img_name_list]
        self.original_annolist = [os.path.join(self._base_dir+'/GlaS_CRAG_cn_npy/train/', 'label', name) for name in self.img_name_list]
        self.num = len(self.original_imglist)
        ################################
        self.scales = self.args.scale
        self.count = 0
        self.bs = self.args.batch_size
        # self.permutation = np.load('permutation_'+str(args.N)+'.npy') ###fixed
        # self.image_list = [self.original_imglist[i] for i in self.permutation]
        # self.anno_list = [self.original_annolist[i] for i in self.permutation]
        self.image_list = self.original_imglist
        self.anno_list = self.original_annolist
        self.image_mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            RandomRotationWithMask(45, resample=False, expand=False, center=None),
            ElasticTransformations(2000, 60),
            torchvision.transforms.ToTensor()
        ])
        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor()
        ])

        if self.num is not None:
            self.image_list = self.image_list[:self.num]
            self.anno_list = self.anno_list[:self.num]#####偶数
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        anno_name = self.anno_list[idx]
        # print(image_name)
        # image = Image.open(image_name)
        # label = Image.open(anno_name)
        image = np.load(image_name)
        label = np.load(anno_name)
        # target = Image.open(self._base_dir + '/GlaS/'+'testA_35.bmp')
        if self.count % self.bs == 0:
            # sf = random.choice([1,.9,.8,.7,.6])
            self.sf = random.choice(self.scales)
            # self.count = 0  # optional
            self.patch_size = math.ceil(self.sf * self.H_size)
        self.count += 1
        # target = target.resize((self.patch_size, self.patch_size), Image.ANTIALIAS)
        # target = np.array(target)

        # image = image/255
        H, W, _ = image.shape
        # print(H, W, self.patch_size)
        ######
        if H < 800 and W < 800 and self.patch_size >= 600: #Glas dataset and 600pix
            image, label = self.pad_image(image, label)
            # print(image.shape)
            image = cv2.resize(image, (800, 600))
            label = cv2.resize(label, (800, 600))
            rnd_h = random.randint(0, max(0, 600 - self.patch_size))
            rnd_w = random.randint(0, max(0, 800 - self.patch_size))
            patchH = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patchL = label[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        else:## < 600pix and CRAG dataset, Glasdataset
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patchH = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patchL = label[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        ###### augmentation - flip, rotate for patch and its annotation
        mode = np.random.randint(0, 8)
        patch_H = util.augment_img(patchH, mode=mode)
        patch_L = util.augment_img(patchL, mode=mode)
        # print(patch_H.shape)
        patch_L[patch_L > 0] = 1
        # print(np.max(patch_L))
        contour_mask = self.obtatin_boundray(patch_L)
        contour_mask = self.dilation(contour_mask, radius=3)
        # patch_H = np.float32(patch_H / 255.)
        patch_H = torch.from_numpy(np.ascontiguousarray(patch_H)).permute(2, 0, 1).float()
        patch_L = torch.from_numpy(np.ascontiguousarray(patch_L)).long().unsqueeze(0)
        contour_mask = torch.from_numpy(np.ascontiguousarray(contour_mask)).long().unsqueeze(0)
        # norm_patch_H = torch.from_numpy(np.ascontiguousarray(norm_patch_H)).permute((2, 0, 1)).float()
        # patch_L, patch_H = self.apply_data_augmentation(patch_H, patch_L)
        # patch = torch.from_numpy(patch_H).float() / 255
        # n_glands = patch_L.max()
        # label = torch.from_numpy(patch_L).float()
        ######
        # patch_L = patch_L.unsqueeze(dim=0)
        sample = {'norm': patch_H, 'label': patch_L, 'contour': contour_mask}
        return sample

    def dilation(self, x, radius=3):
        """ Return greyscale morphological dilation of an image,
        see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`_.

        Parameters
        -----------
        x : 2D array image.
        radius : int for the radius of mask.
        """
        from skimage.morphology import disk, dilation
        mask = disk(radius)
        x = dilation(x, footprint=mask)
        return x

    def obtatin_boundray(self, gt_mask):
        contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros_like(gt_mask)
        mask2 = np.zeros_like(np.ascontiguousarray(gt_mask))
        cv2.drawContours(mask2, contours, -1, (255, 0, 0), 1)
        # n = len(contours)
        # c = contours[0]
        # c = np.squeeze(c)
        # for i in range(n-1):
        #     c_i = contours[i+1]
        #     c_i = np.squeeze(c_i)
        #     c = np.concatenate((c, c_i), axis=0)
        # x = c[:, 0]
        # y = c[:, 1]
        # mask[y, x] = 255
        return mask2/255.0


    def color_normalization(self, target, to_transform):
        # target = staintools.read_image("./data/my_target_image.png")
        # to_transform = staintools.read_image("./data/my_image_to_transform.png")

        # Standardize brightness (optional, can improve the tissue mask calculation)
        target = staintools.LuminosityStandardizer.standardize(target)
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

        # Stain normalize
        normalizer = staintools.StainNormalizer(method='reinhard')
        normalizer.fit(target)
        transformed = normalizer.transform(to_transform)
        return transformed

    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

            # we pad more than needed to later do translation augmentation
        pad_h = max((600 - image.shape[0]) // 2, 0)
        pad_w = max((800 - image.shape[1]) // 2, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def normalize_img(self, img):
        if np.max(img) != np.min(img):
            trans_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            trans_img = img
        return trans_img

    def prepocess(self, img):
        return pre(img)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)