import os
import torch
import numpy as np
import scipy
import torchvision
import itertools
import cv2
from dataloaders.augmentations import ElasticTransformations, RandomRotationWithMask
from torch.utils.data import Dataset
from PIL import Image
from cd_Utils import Preprocess as pre
import utils.util as util
# import itertools
# import staintools
import math
import random
from torch.utils.data.sampler import Sampler
cv2.setNumThreads(0)


class Gland_ACWE_dataset(Dataset):##GlaS dataset and CRAG dataset
    """ LA Dataset """
    def __init__(self, args, fixed_patch_size=(416, 416), base_dir=None, local_rank=0, validation=False):
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
        self.fix_ed_patch_size = fixed_patch_size
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
        if H < 800 and W < 800 and self.patch_size == 600 or self.sf ==0.8: #Glas dataset and 600pix
            image, mask = self.pad_image(image, mask)
            # print(image.shape)
            image = cv2.resize(image, (775, 600))
            mask = cv2.resize(mask, (775, 600))
            rnd_h = random.randint(0, max(0, 600 - self.patch_size))
            rnd_w = random.randint(0, max(0, 775 - self.patch_size))
            patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        elif self.patch_size == 1200:## Glas dataset
            if H < 800 and W < 800:
                image = cv2.resize(image, (775, 522))
                mask = cv2.resize(mask, (775, 522))
                patch_H = image
                patch_L = mask
            else:
                rnd_h = random.randint(0, max(0, H - 522))
                rnd_w = random.randint(0, max(0, W - 775))
                patch_H = image[rnd_h:rnd_h + 522, rnd_w:rnd_w + 775, :]
                patch_L = mask[rnd_h:rnd_h + 522, rnd_w:rnd_w + 775]
        elif self.patch_size == 800:
            if H < 800 and W < 800:
                image, mask = self.pad_image800(image, mask)
                image = cv2.resize(image, (800, 800))
                mask = cv2.resize(mask, (800, 800))
                patch_H = image
                patch_L = mask
            else:
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
                patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
                patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        else:## < 600pix and CRAG dataset, Glasdataset
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]

        ###### augmentation - flip, rotate for patch and its annotation
        patch_H = util.augment_img(patch_H, mode=self.mode)
        patch_L = util.augment_img(patch_L, mode=self.mode)
        # print(patch_H.shape)
        patch_L, patch_H = self.apply_data_augmentation(patch_H, patch_L)
        ######
        # contour_mask = self.obtatin_boundray(np.ascontiguousarray(patch_L))
        # contour_mask = self.dilation(contour_mask, radius=3)
        label_L = self.create_eroded_mask(patch_L)
        trans_Label = label_L[1].numpy().astype(np.uint8)
        contour_mask = self.obtatin_boundray(trans_Label)
        contour_mask = self.dilation(contour_mask, radius=3)
        contour_mask = torch.from_numpy(contour_mask).unsqueeze(0)
        sample = {'image': patch_H, 'label': label_L, 'contour': contour_mask}
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
        mask2 = np.zeros_like(np.ascontiguousarray(gt_mask))
        cv2.drawContours(mask2, contours, -1, (255, 0, 0), 1)
        return mask2 / 255.0


    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

            # we pad more than needed to later do translation augmentation
        pad_h = max((self.H_size - image.shape[0]) // 2, 0)
        pad_w = max((self.H_size - image.shape[1]) // 2, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def pad_image800(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

            # we pad more than needed to later do translation augmentation
        pad_h = max((800 - image.shape[0]) // 2, 0)
        pad_w = max((800 - image.shape[1]) // 2, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def apply_data_augmentation(self, image, mask):
        """Helper function to apply all configured data augmentations on both mask and image"""
        img = np.ascontiguousarray(image).transpose(2, 0, 1)
        # b_mask = (mask > 0).astype(np.uint8)
        # print(b_mask)
        # contour_mask = self.obtatin_boundray(np.ascontiguousarray(b_mask))
        patch = torch.from_numpy(img).float() / 255
        n_glands = mask.max()
        label = torch.from_numpy(np.ascontiguousarray(mask)).float() / n_glands if n_glands > 0 \
            else torch.from_numpy(np.ascontiguousarray(mask))
        # contour_mask = torch.from_numpy(contour_mask).float()
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
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomVerticalFlip(),
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
        mask = np.load(anno_name)
        # target = Image.open(self._base_dir + '/GlaS/'+'testA_35.bmp')
        if self.count % self.bs == 0:
            # sf = random.choice([1,.9,.8,.7,.6])
            self.sf = random.choice(self.scales)
            # self.count = 0  # optional

            self.patch_size = math.ceil(self.sf * self.H_size)
            self.mode = np.random.randint(0, 8)
        self.count += 1
        # image = image/255
        H, W, _ = image.shape
        # print(H, W, self.patch_size)
        ######
        if H < 800 and W < 800 and self.patch_size == 600 or self.sf ==0.8: #Glas dataset and 600pix
            image, mask = self.pad_image(image, mask)
            # print(image.shape)
            image = cv2.resize(image, (775, 600))
            mask = cv2.resize(mask, (775, 600))
            rnd_h = random.randint(0, max(0, 600 - self.patch_size))
            rnd_w = random.randint(0, max(0, 775 - self.patch_size))
            patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        elif self.patch_size == 1200:## Glas dataset
            if H < 800 and W < 800:
                image = cv2.resize(image, (775, 522))
                mask = cv2.resize(mask, (775, 522))
                patch_H = image
                patch_L = mask
            else:
                rnd_h = random.randint(0, max(0, H - 522))
                rnd_w = random.randint(0, max(0, W - 775))
                patch_H = image[rnd_h:rnd_h + 522, rnd_w:rnd_w + 775, :]
                patch_L = mask[rnd_h:rnd_h + 522, rnd_w:rnd_w + 775]
        elif self.patch_size == 800:
            if H < 800 and W < 800:
                image, mask = self.pad_image800(image, mask)
                image = cv2.resize(image, (800, 800))
                mask = cv2.resize(mask, (800, 800))
                patch_H = image
                patch_L = mask
            else:
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
                patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
                patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        else:## < 600pix and CRAG dataset, Glasdataset
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]

        ###### augmentation - flip, rotate for patch and its annotation
        patch_H = util.augment_img(patch_H, mode=self.mode)
        patch_L = util.augment_img(patch_L, mode=self.mode)
        # print(np.max(patch_H))
        patch_L, patch_H = self.apply_data_augmentation(patch_H, patch_L)
        label_L = self.create_eroded_mask(patch_L)
        # print(label_L.shape)
        trans_Label = label_L[1].numpy().astype(np.uint8)
        contour_mask = self.obtatin_boundray(trans_Label)
        contour_mask = self.dilation(contour_mask, radius=3)
        contour_mask = torch.from_numpy(np.ascontiguousarray(contour_mask)).long().unsqueeze(0)
        inner_mask = label_L[1] * (1-contour_mask).float()
        sample = {'norm': patch_H, 'label': label_L, 'contour': contour_mask, 'inner_mask': inner_mask}
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
        return mask2/255.0

    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

        # we pad more than needed to later do translation augmentation
        pad_h = max((self.H_size - image.shape[0]) // 2, 0)
        pad_w = max((self.H_size - image.shape[1]) // 2, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def pad_image800(self, image, mask):
        """Helper function to pad smaller image to the correct size"""

            # we pad more than needed to later do translation augmentation
        pad_h = max((800 - image.shape[0]) // 2, 0)
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

    def apply_data_augmentation(self, image, mask):
        """Helper function to apply all configured data augmentations on both mask and image"""
        img = np.ascontiguousarray(image).transpose(2, 0, 1)
        # b_mask = (mask > 0).astype(np.uint8)
        # print(b_mask)
        # contour_mask = self.obtatin_boundray(np.ascontiguousarray(b_mask))
        patch = torch.from_numpy(img).float() if np.max(img) <= 1 else torch.from_numpy(img).float()/np.max(img)
        n_glands = mask.max()
        label = torch.from_numpy(np.ascontiguousarray(mask)).float() / n_glands if n_glands > 0 \
            else torch.from_numpy(np.ascontiguousarray(mask))
        # contour_mask = torch.from_numpy(contour_mask).float()
        # print(patch.shape, label.shape)
        if not self.validation:
            patch_label_concat = torch.cat((patch, label[None, :, :].float()))
            patch_label_concat = self.image_mask_transforms(patch_label_concat)
            patch, label = patch_label_concat[0:3], np.round(patch_label_concat[3] * n_glands)
            # patch = self.image_transforms(patch)
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