# Utils used with tensorflow implemetation
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import copy
from PIL import Image
# from utils.Utils import SeparateStains
import matplotlib.pylab as plt
import scipy.misc as misc
# from torchvision import transforms
def preprocess1(input):
    #input = copy.deepcopy(valid_images)
    input = copy.deepcopy(np.float64(input))
    s = input.shape
    if len(s) == 4:
        for i in range(s[0]):
            input1 = copy.deepcopy(input[i, :, :, :])
            for j in range(s[-1]):
                b = copy.deepcopy(input1[:, :, j])
                b1 = b - np.mean(b)
                b = copy.deepcopy(b1)
                input1[:, :, j] = copy.deepcopy(b)
            input[i, :, :, :] = copy.deepcopy(input1)
    else:
        input1 = copy.deepcopy(input)
        for j in range(s[-1]):
            b = copy.deepcopy(input1[:, :, j])
            b1 = b - np.mean(b)
            b = copy.deepcopy(b1)
            input1[:, :, j] = copy.deepcopy(b)
        input = copy.deepcopy(input1)
    return input
def Preprocess(inputi):
    #input = copy.deepcopy(valid_images)
    s = inputi.shape
    # print(s)
    if len(s) == 4:
        input = np.zeros([s[0], s[1], s[2], s[3]])
        # for i in range(s[0]):
        #     input1 = copy.deepcopy(input[i, :, :, :])
        #     for j in range(s[-1]):
        #         b = copy.deepcopy(input1[:, :, j])
        #         b1 = b - np.mean(b)
        #         b = copy.deepcopy(b1)
        #         input1[:, :, j] = copy.deepcopy(b)
        #     input[i, :, :, :] = copy.deepcopy(input1)
        ele = 1
        #imageURL = 'D:/redo3/images/training/1972.png'
        #b1 = misc.imread('D:/redo3/annotations/training/1972.png')
        #imageRGB = misc.imread(imageURL)
        He = np.array([0.644211, 0.716556, 0.266844], 'float64')
        Eo = np.array([0.092789, 0.954111, 0.283111], 'float64')
        Res = np.array([0, 0, 0], 'float64')
        for i in range(3):
            if He[i]**2+Eo[i]**2 > 1:
                Res[i] = 0.001
            else:
                Res[i] = np.sqrt(1-He[i]**2-Eo[i]**2)
        HDABtoRGB = np.matrix([He/np.linalg.norm(He), Eo/np.linalg.norm(Eo), Res/np.linalg.norm(Res)])
        RGBtoHDAB = np.linalg.inv(HDABtoRGB)
        for j in range(s[0]):
            k = 0
            k = k + 1
            input1 = copy.deepcopy(inputi[j, :, :, :])
            imageHDAB = SeparateStains(input1, RGBtoHDAB)
            #misc.imsave('D:\\'+str(k)+'.png', imageHDAB[:, :, 0])
            cc = np.zeros([s[1], s[2], s[3]])
            c = copy.deepcopy(imageHDAB[:, :, 0])
            if c.max() != c.min():
               c = np.round(255*(c-c.min())/(c.max()-c.min()))
            else:
               c = np.round(255 + c-c.min())
            c = copy.deepcopy(np.array(c, 'uint8'))
            for i in range(3):
                cc[:, :, i] = copy.deepcopy(c)
            #misc.imsave('E:\\1.png', cc)
            #cc1 = misc.imread('E:\\1.png')
            input[j, :, :, :] = copy.deepcopy(cc)
        input = preprocess1(input)
    if len(s) == 3:
        # for i in range(s[0]):
        #     input1 = copy.deepcopy(input[i, :, :, :])
        #     for j in range(s[-1]):
        #         b = copy.deepcopy(input1[:, :, j])
        #         b1 = b - np.mean(b)
        #         b = copy.deepcopy(b1)
        #         input1[:, :, j] = copy.deepcopy(b)
        #     input[i, :, :, :] = copy.deepcopy(input1)
        ele = 1
        # imageURL = 'D:/redo3/images/training/1972.png'
        # b1 = misc.imread('D:/redo3/annotations/training/1972.png')
        # imageRGB = misc.imread(imageURL)
        He = np.array([0.644211, 0.716556, 0.266844], 'float64')
        Eo = np.array([0.092789, 0.954111, 0.283111], 'float64')
        Res = np.array([0, 0, 0], 'float64')
        for i in range(3):
            if He[i] ** 2 + Eo[i] ** 2 > 1:
                Res[i] = 0.001
            else:
                Res[i] = np.sqrt(1 - He[i] ** 2 - Eo[i] ** 2)
        HDABtoRGB = np.matrix([He / np.linalg.norm(He), Eo / np.linalg.norm(Eo), Res / np.linalg.norm(Res)])
        RGBtoHDAB = np.linalg.inv(HDABtoRGB)
        input1 = copy.deepcopy(inputi)
        # print(input1.shape, RGBtoHDA.shape)
        imageHDAB = SeparateStains(input1, RGBtoHDAB)
        # misc.imsave('D:\\'+str(k)+'.png', imageHDAB[:, :, 0])
        cc = np.zeros([s[0], s[1], s[2]])
        c = copy.deepcopy(imageHDAB[:, :, 0])
        if c.max() != c.min():
            c = np.round(255 * (c - c.min()) / (c.max() - c.min()))
        else:
            c = np.round(255 + c - c.min())
        c = copy.deepcopy(np.array(c, 'uint8'))
        for i in range(3):
            cc[:, :, i] = copy.deepcopy(c)
        # misc.imsave('E:\\1.png', cc)
        # cc1 = misc.imread('E:\\1.png')
        input = preprocess1(cc)
    return input

def resize_img(img_pil, scale, type, resize_size):
    """
    input: PIL Image
    return: PIL Image
    """
    w = img_pil.size[0]
    h = img_pil.size[1]
    if not resize_size:
        nw = int(w * scale)
        nh = int(h * scale)
    else:
        nw, nh = resize_size
    if type == "image":
        img_pil = img_pil.resize((nw, nh), Image.ANTIALIAS)
    elif type == "label":
        img_pil = img_pil.resize((nw, nh), Image.NEAREST)
    else:
        img_pil = img_pil.resize((nw, nh), Image.ANTIALIAS)
    return img_pil

def crop_img(im, size):
    """
    crop image
    """
    w = im.size[0]
    h = im.size[1]
    return im.crop(map(int, [w*(0.5-0.5/size), h*(0.5-0.5/size), w*(0.5+0.5/size), h*(0.5+0.5/size)]))

def trans(image, normalize):
    '''
    input: numpy.narray
    '''
    image_out = transforms.ToTensor()(image)
    if normalize:
        image_out = transforms.Normalize(
            normalize["mean"],
            normalize["std"]
            )(image_out)
    return image_out.numpy()

def SeparateStains(imageRGB, matrix):
   imageRGB = np.float64(imageRGB)+2
   imagesize = imageRGB.shape
   # print(imagesize)
   B = np.reshape(-np.log(imageRGB/257), [imagesize[0]*imagesize[1], 3], 'F')
   # print(B.shape)
   # print(matrix)
   for i in range(imagesize[0]*imagesize[1]):
       if np.sum(B[i, :])/3 < 0.25 or B[i, 0] < 0.2 or B[i, 1] < 0.2 or B[i, 2] < 0.2:
           B[i, :] = 0
   imageout = np.matmul(B, matrix)
   imageout[imageout < 0] = 0
   imageout = np.reshape(np.array(imageout), [imagesize[0], imagesize[1], 3], 'F')
   return imageout