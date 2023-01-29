import os
import argparse
import torch
import numpy as np
import scipy
import cv2
import skimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
# from node.metrics import ObjectDice, ObjectHausdorff, F1score
from networks.unet_model import UNet_UAMT
from networks.unet_model import UNet_UAMT
from networks.unet_model import UNet
from node.metrics import ObjectDice, ObjectHausdorff, F1score
from cd_Utils import Preprocess as pre
from PIL import Image
from networks.unet_model import UNet2
from medpy import metric
# from dataloaders.data_load_ACWE import Gland_ACWE_original_img_loader as gland_set
import matplotlib.pyplot as plt
import matplotlib
# # matplotlib.use('Tkagg')
# matplotlib.use('QT5agg')
from test_one_image import predict_softmax


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='../../datasets/Glas/testA/images/testA_1crop_5.png',
                        help='the path of image')
    parser.add_argument('--gt_mask_path', type=str, default='../../datasets/Glas/testA/anno/testA_1crop_5.png',
                        help='the path of image')
    parser.add_argument('--model', type=str, default='UNet_ours', help='model_name')
    parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
    parser.add_argument('--t', type=float, default=0.559, help='threshold for ACWE')
    # parser.add_argument('--iter', type=int,  default=6000, help='model iteration')
    parser.add_argument('--detail', type=int, default=0, help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
    return parser.parse_args()


def trans(x):
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.78780321, 0.5120167, 0.78493782], [0.16766301, 0.24838048, 0.13225162])
    ])
    return image_transforms(x)

def obtatin_boundray2(gt_mask):
    contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros_like(gt_mask)
    mask2 = np.zeros_like(np.ascontiguousarray(gt_mask))
    cv2.drawContours(mask2, contours, -1, (255, 0, 0), 1)
    return mask2/255.0

def calculate_metric_perimg(pred, gt):
    f1_img = F1score(pred, gt)
    hausdorff_img = ObjectHausdorff(pred, gt)
    dice_img = ObjectDice(pred, gt)
    # print(dice_img, f1_img, hausdorff_img, dice, jc, hd, asd)
    return dice_img, f1_img, hausdorff_img

def calculate_pixel_dice_jc_hd_perimg(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    return dice, jc, hd

def calculate_imglist_metric(net, device, imglist, masklist, args):
    total_metric = 0.0
    # test_path = args.test_root_path
    # mask_path = args.test_mask_path
    # image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for (name, maskname) in zip(imglist, masklist):
        print(name)
        img = Image.open(name)
        gt_mask = Image.open(maskname)
        img = img.resize((775, 522), Image.ANTIALIAS)
        gt_mask = gt_mask.resize((775, 522), Image.ANTIALIAS)
        # img
        gt_mask = np.array(gt_mask)

        # print(image.shape, label.shape)
        results = inference_image(net, img, device)
        pred_mask = postprocess(results, args.t)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num +=1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def test_calculate_metric_two_channel(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = Image.open(test_path+name)
        gt_mask = Image.open(mask_path+name)
        # img
        # gt_mask = np.array(gt_mask)
        # print(image.shape, label.shape)
        # gt_mask[gt_mask > 0] = 1
        gt_mask = skimage.measure.label(np.array(gt_mask))
        # img = img.resize((400, 400), Image.ANTIALIAS)
        # name1 = name[:-4]
        img = np.array(img)
        pred_mask = predict_softmax(net, img, device)
        # from inference_one_img import postprocess3
        pred_mask = postprocess3(pred_mask)
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num +=1
            print(num)
            print(single_metric)
        total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def calculate_all_metric(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = Image.open(test_path+name)
        # img = trans(img)
        gt_mask = Image.open(mask_path+name)
        # img
        # gt_mask = np.array(gt_mask)
        # print(image.shape, label.shape)
        results = inference_image(net, img, device, test_aug=False)
        pred_mask = postprocess(results, args.t)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def calculate_all_metric_cn(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = np.load(test_path+name)
        # img = img.transpose((2, 0, 1))/255.0
        # img = torch.from_numpy(np.ascontiguousarray(img)).float().permute((2, 0, 1))
        # gt_mask = Image.open(mask_path+name)
        # img
        gt_mask = np.load(mask_path+name)
        # print(image.shape, label.shape)
        results = inference_image(net, img, device, test_aug=False)
        pred_mask = postprocess(results)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric


def calculate_all_metric_with_normalization(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = Image.open(test_path+name)
        # img = trans(img)
        img = trans(img)
        gt_mask = Image.open(mask_path+name)
        # img
        gt_mask = np.array(gt_mask)
        # print(image.shape, label.shape)
        results = inference_image(net, img, device, test_aug=False)
        pred_mask = postprocess(results, args.t)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def calculate_SRNet_metric(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = Image.open(test_path+name)
        # img = trans(img)
        gt_mask = Image.open(mask_path+name)
        # img
        gt_mask = np.array(gt_mask)
        # print(image.shape, label.shape)
        results = inference_image2(net, img, device)
        pred_mask = postprocess2(results, args.t)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric


def calculate_net_one_channel_metric(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = Image.open(test_path+name)
        # img = trans(img)
        gt_mask = Image.open(mask_path+name)
        # img
        gt_mask = np.array(gt_mask)
        # print(image.shape, label.shape)
        results = inference_net_one_channel_image(net, img, device)
        pred_mask = postprocess2(results, args.t)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def calculate_net_one_channel_cn_metric(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = Image.open(test_path+name)
        # img = trans(img)
        img = np.array(img)
        img = pre(img)
        img = normalize_img(img)
        # img = img.transpose((2, 0, 1))/255.0
        img = torch.from_numpy(np.ascontiguousarray(img)).float().permute((2, 0, 1))

        gt_mask = Image.open(mask_path+name)
        # img
        gt_mask = np.array(gt_mask)

        # print(image.shape, label.shape)
        pred_mask = predict_one_channel(net, img, device)
        pred_mask = postprocess3(pred_mask)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def calculate_net_one_channel_cn_metric2(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = np.load(test_path+name)
        # img = img.transpose((2, 0, 1))/255.0
        # img = torch.from_numpy(np.ascontiguousarray(img)).float().permute((2, 0, 1))
        # gt_mask = Image.open(mask_path+name)
        # img
        gt_mask = np.load(mask_path+name)
        # print(image.shape, label.shape)
        pred_mask = evaluate_net_one_channel_cn_image(net, img, device, test_time_augmentation=False)
        pred_mask = postprocess2(pred_mask)
        gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_metric_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def calculate_net_one_channel_contour_metric(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = Image.open(test_path+name)
        # img = trans(img)
        gt_mask = Image.open(mask_path+name)
        # img
        gt_mask = np.array(gt_mask)
        from test_one_image import dilation, obtatin_boundray
        gt_mask = obtatin_boundray2(gt_mask)
        gt_mask = dilation(gt_mask, radius=3)
        # print(image.shape, label.shape)
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img / 255
        img = torch.from_numpy(img).float().contiguous()
        pred_mask = predict_one_channel(net, img, device)
        # results = inference_net_one_channel_image(net, img, device)
        # pred_mask = postprocess2(results, args.t)
        # gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_pixel_dice_jc_hd_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def calculate_net_one_channel_contour_cn_metric(net, device, args):
    total_metric = 0.0
    test_path = args.test_root_path
    mask_path = args.test_mask_path
    image_list = os.listdir(test_path)
    # print("init weight from {}".format(save_mode_path))
    net.eval()
    num=0
    for name in image_list:
        print(name)
        img = np.load(test_path+name)
        # img = trans(img)
        gt_mask = np.load(mask_path+name)
        # img
        # gt_mask = np.array(gt_mask)
        from test_one_image import dilation, obtatin_boundray
        gt_mask = obtatin_boundray2(gt_mask)
        gt_mask = dilation(gt_mask, radius=3)
        # print(image.shape, label.shape)
        # img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img / 255 if np.max(img) > 1 else img
        img = torch.from_numpy(img).float().contiguous()
        pred_mask = predict_one_sigmoid_channel(net, img, device)
        # results = inference_net_one_channel_image(net, img, device)
        # pred_mask = postprocess2(results, args.t)
        # gt_mask = skimage.measure.label(np.array(gt_mask))
        if ~np.all(gt_mask==0) and ~np.all(pred_mask==0):
            single_metric = calculate_pixel_dice_jc_hd_perimg(pred_mask, gt_mask)
            num += 1
            print(num)
            print(single_metric)
            total_metric += np.asarray(single_metric)
    avg_metric = total_metric / num
    return avg_metric

def normalize_img(img):
    if np.max(img)!= np.min(img):
        trans_img = (img-np.min(img))/(np.max(img)-np.min(img))
        return trans_img
    else:
        # trans_img
        return img


def dilation(x, radius=3):
    """ Return greyscale morphological dilation of an image,
    see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`_.

    Parameters
    -----------
    x : 2D array image.
    radius : int for the radius of mask.
    """
    from skimage.morphology import disk, dilation
    mask = disk(radius)
    x = dilation(x, selem=mask)
    return x

def inference_image(net, image, device, test_aug=True):
    """Helper function to inference an PIL image with a net, shouldpad in the case of U-Net"""
    if type(image) is not np.ndarray:
        image = np.array(image)
    result = evaluate_image(net, image, device, test_time_augmentation=test_aug)
    return result

def inference_image2(net, image, device):
    """Helper function to inference an PIL image with a net, shouldpad in the case of U-Net"""
    image = np.array(image)
    result = evaluate_image2(net, image, device)
    return result

def inference_net_one_channel_image(net, image, device):
    """Helper function to inference an PIL image with a net, shouldpad in the case of U-Net"""
    image = np.array(image)
    result = evaluate_net_one_channel_image(net, image, device)
    return result

def postprocess(result, threshold1=0.5):
    """Helper function postprocess inference_image result of GlaS challenge"""
    splitted = split_objects(result, threshold1)
    labeled = skimage.measure.label(np.array(splitted))
    temp = remove_small_object(labeled, threshold=500)
    growed = grow_to_fill_borders(temp, result[1] > threshold1)
    hole_filled = hole_filling_per_object(growed)
    final = remove_small_object(hole_filled, threshold=500)
    return final

def postprocess2(result, threshold1=0.5):
    """Helper function postprocess inference_image result of GlaS challenge"""
    splitted = split_objects(result, threshold1)
    labeled = skimage.measure.label(np.array(splitted))
    temp = remove_small_object(labeled, threshold=500)
    # growed = grow_to_fill_borders(temp, result[1] > threshold1)
    hole_filled = hole_filling_per_object(temp)
    final = remove_small_object(hole_filled, threshold=500)
    return final

def postprocess3(result):
    """Helper function postprocess inference_image result of GlaS challenge"""
    # splitted = split_objects(result, threshold1)
    labeled = skimage.measure.label(np.array(result))
    temp = remove_small_object(labeled, threshold=500)
    # growed = grow_to_fill_borders(temp, result[1] > threshold1)
    hole_filled = hole_filling_per_object(temp)
    final = remove_small_object(hole_filled, threshold=500)
    return final

def evaluate_image(net, image, device, test_time_augmentation=True):
    """
    Helper function to inference a numpy matrix with a net and optionally
    Args:
        net (nn.Module): the neural network
        image (np.array): the image
        test_time_augmentation: (bool): whether to apply test-time-augmentation (averaging across three flips)
        shouldpad (bool): whether to reflect pad the image so that
            the output of U-Net is equal to input size
    Returns:
        np.array: neural network prediction
    """
    net.eval()
    with torch.no_grad():
        def _eval_img(img):
            torch_image = torch.from_numpy(img).float()
            _, result = net(torch_image[None].to(device))
            soft_result = torch.sigmoid(result)[0].cpu()
            soft_result_np = soft_result.detach().numpy().transpose(1, 2, 0)
            return soft_result_np

        transposed_image = image.transpose(2, 0, 1) / 255
        soft_result_np = _eval_img(transposed_image)
        if not test_time_augmentation:
            transposed_image_ud = np.flipud(image).transpose(2, 0, 1) / 255
            transposed_image_lr = np.fliplr(image).transpose(2, 0, 1) / 255
            soft_result_ud = _eval_img(transposed_image_ud)
            soft_result_lr = _eval_img(transposed_image_lr)
            soft_result_np_ud = np.flipud(soft_result_ud)
            soft_result_np_lr = np.fliplr(soft_result_lr)
            soft_result_np = (soft_result_np + soft_result_np_ud + soft_result_np_lr) / 3
        else:
            from utils.util import augment_img, inv_augment_img
            for i in range(8):
                transposed_img = augment_img(image, mode=i+1)
                result = _eval_img(transposed_img.transpose(2, 0, 1) / 255)
                soft_result_np += inv_augment_img(result, mode=i+1)
            soft_result_np = soft_result_np/8
        return soft_result_np.transpose(2, 0, 1)

def evaluate_net_one_channel_image(net, image, device, test_time_augmentation=False):
    """
    Helper function to inference a numpy matrix with a net and optionally
    Args:
        net (nn.Module): the neural network
        image (np.array): the image
        test_time_augmentation: (bool): whether to apply test-time-augmentation (averaging across three flips)
        shouldpad (bool): whether to reflect pad the image so that
            the output of U-Net is equal to input size
    Returns:
        np.array: neural network prediction
    """
    net.eval()
    with torch.no_grad():
        def _eval_net_one_channel_img(img):
            torch_image = torch.from_numpy(img).float()
            soft_result = net(torch_image[None].to(device))[0].cpu()
            # soft_result = torch.sigmoid(result)[0].cpu()
            soft_result_np = soft_result.detach().numpy().transpose(1, 2, 0)
            return soft_result_np

        transposed_image = image.transpose(2, 0, 1) / 255
        soft_result_np = _eval_net_one_channel_img(transposed_image)
        if test_time_augmentation:
            transposed_image_ud = np.flipud(image).transpose(2, 0, 1) / 255
            transposed_image_lr = np.fliplr(image).transpose(2, 0, 1) / 255
            soft_result_ud = _eval_net_one_channel_img(transposed_image_ud)
            soft_result_lr = _eval_net_one_channel_img(transposed_image_lr)
            soft_result_np_ud = np.flipud(soft_result_ud)
            soft_result_np_lr = np.fliplr(soft_result_lr)
            soft_result_np = (soft_result_np + soft_result_np_ud + soft_result_np_lr) / 3
        else:
            from utils.util import augment_img, inv_augment_img
            for i in range(8):
                transposed_img = augment_img(image, mode=i+1)
                result = _eval_net_one_channel_img(transposed_img.transpose(2, 0, 1) / 255)
                soft_result_np += inv_augment_img(result, mode=i+1)
            soft_result_np = soft_result_np/8
        return soft_result_np.transpose(2, 0, 1)

def evaluate_net_one_channel_cn_image(net, image, device, test_time_augmentation=True):
    net.eval()
    with torch.no_grad():
        def _eval_net_one_channel_cn_img(img):
            torch_image = torch.from_numpy(np.ascontiguousarray(img)).float()
            result = net(torch_image[None].to(device)).cpu()
            soft_result = torch.sigmoid(result)[0].cpu()
            soft_result_np = soft_result.detach().numpy().transpose(1, 2, 0)
            return soft_result_np

        transposed_image = image.transpose(2, 0, 1)
        soft_result_np = _eval_net_one_channel_cn_img(transposed_image)
        if not test_time_augmentation:
            transposed_image_ud = np.flipud(image).transpose(2, 0, 1)
            transposed_image_lr = np.fliplr(image).transpose(2, 0, 1)
            soft_result_ud = _eval_net_one_channel_cn_img(transposed_image_ud)
            soft_result_lr = _eval_net_one_channel_cn_img(transposed_image_lr)
            soft_result_np_ud = np.flipud(soft_result_ud)
            soft_result_np_lr = np.fliplr(soft_result_lr)
            soft_result_np = (soft_result_np + soft_result_np_ud + soft_result_np_lr) / 3
        else:
            from utils.util import augment_img, inv_augment_img
            for i in range(8):
                transposed_img = augment_img(image, mode=i+1)
                result = _eval_net_one_channel_cn_img(transposed_img.transpose(2, 0, 1))
                soft_result_np += inv_augment_img(result, mode=i+1)
            soft_result_np = soft_result_np/8
        return soft_result_np.transpose(2, 0, 1)

def predict_one_channel(net, img, device, threshold=0.5):
    img = img.unsqueeze(dim=0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        # _, outputs = net(img)
        outputs_soft = net(img)
    result = outputs_soft.squeeze(dim=0).squeeze(dim=0).cpu()
    mask = (result >= threshold).numpy()
    return mask

def predict_one_sigmoid_channel(net, img, device, threshold=0.5):
    img = img.unsqueeze(dim=0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        # _, outputs = net(img)
        outputs_soft = torch.sigmoid(net(img))

    result = outputs_soft.squeeze(dim=0).squeeze(dim=0).cpu()
    mask = (result >= threshold).numpy()
    return mask

def evaluate_image2(net, image, device, test_time_augmentation=True):
    """
    Helper function to inference a numpy matrix with a net and optionally
    Args:
        net (nn.Module): the neural network
        image (np.array): the image
        test_time_augmentation: (bool): whether to apply test-time-augmentation (averaging across three flips)
        shouldpad (bool): whether to reflect pad the image so that
            the output of U-Net is equal to input size
    Returns:
        np.array: neural network prediction
    """
    net.eval()
    with torch.no_grad():
        def _eval_img2(img):
            torch_image = torch.from_numpy(img).float()
            x_seg, x_res = net(torch_image[None].to(device))
            soft_result = x_seg+x_res
            soft_result_np = soft_result.detach().numpy().transpose(1, 2, 0)
            return soft_result_np

        transposed_image = image.transpose(2, 0, 1) / 255
        soft_result_np = _eval_img2(transposed_image)
        if test_time_augmentation:
            transposed_image_ud = np.flipud(image).transpose(2, 0, 1) / 255
            transposed_image_lr = np.fliplr(image).transpose(2, 0, 1) / 255
            soft_result_ud = _eval_img2(transposed_image_ud)
            soft_result_lr = _eval_img2(transposed_image_lr)
            soft_result_np_ud = np.flipud(soft_result_ud)
            soft_result_np_lr = np.fliplr(soft_result_lr)
            soft_result_np = (soft_result_np + soft_result_np_ud + soft_result_np_lr) / 3
        return soft_result_np.transpose(2, 0, 1)

def split_objects(image, threshold=0.5):
    """Helper function to threshold image and thereby split close glands"""
    return (image[0] > threshold)

def remove_small_object(labeled_image, threshold=500):
    """Helper function to remove small objects"""
    regionprops = skimage.measure.regionprops(labeled_image)
    new_results = np.array(labeled_image).copy()
    for prop in regionprops:
        if prop.area < threshold:
            new_results[new_results == prop.label] = 0
    return new_results

def grow_to_fill_borders(eroded_result, full_result):
    """
    Helper function to use a maximum filter and grow all labeled regions
    constraint to the area of the full prediction.
    """
    for i in range(10):
        new_labeled = scipy.ndimage.maximum_filter(eroded_result, 3)
        eroded_result[full_result == 1] = new_labeled[full_result == 1]
    eroded_result[full_result == 0] = 0
    return eroded_result

def hole_filling_per_object(image):
    """Helper function to fill holes inside individual labeled regions"""
    grow_labeled = image
    for i in np.unique(grow_labeled):
        if i == 0: continue
        filled = scipy.ndimage.morphology.binary_fill_holes(grow_labeled == i)
        grow_labeled[grow_labeled == i] = 0
        grow_labeled[filled == 1] = i
    return grow_labeled

def create_eroded_mask(label):
    """Helper function to create a mask where every gland is eroded"""
    boundaries = np.zeros(label.shape)
    for i in np.unique(label):
        if i == 0: continue  # the first label is background
        gland_mask = (label == i)
        binarized_mask_border = scipy.ndimage.morphology.binary_erosion(gland_mask,
                                                                        structure=np.ones((13, 13)),
                                                                        border_value=1)

        # binarized_mask_border = torch.from_numpy(binarized_mask_border.astype(np.float32))
        boundaries[label == i] = binarized_mask_border[label == i]

    # label = (label > 0).float()
    # label = torch.stack((boundaries, label))
    return boundaries

def predict(net, img, device, threshold=0.5):
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, outputs = net(img)
        outputs_soft = torch.sigmoid(outputs)
    result = outputs_soft.squeeze(dim=0).squeeze(dim=0).cpu()
    mask = (result>=threshold).numpy()
    # mask = np.array(mask)
    mask1 = mask.astype(np.uint8)
    return mask1

def predict2(net, img, device, threshold=0.5):
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, outputs = net(img)
        outputs_soft = torch.sigmoid(outputs)
    result = outputs_soft.squeeze(dim=0).squeeze(dim=0).cpu()
    mask = (result >= threshold).numpy()
    return mask

def predict_softmax(net, image, device):
    with torch.no_grad():
        def _eval_img(img):
            torch_image = torch.from_numpy(img).float()
            result = net(torch_image[None].to(device))
            soft_result = torch.sigmoid(result)[0]
            soft_result = F.softmax(soft_result, dim=1)
            result = soft_result.argmax(dim=0).squeeze(dim=0).cpu().numpy()
            return result

        transposed_image = image.transpose(2, 0, 1) / 255
        soft_result_np = _eval_img(transposed_image)
        transposed_image_ud = np.flipud(image).transpose(2, 0, 1) / 255
        transposed_image_lr = np.fliplr(image).transpose(2, 0, 1) / 255
        soft_result_ud = _eval_img(transposed_image_ud)
        soft_result_lr = _eval_img(transposed_image_lr)
        soft_result_np_ud = np.flipud(soft_result_ud)
        soft_result_np_lr = np.fliplr(soft_result_lr)
        soft_result_np = (soft_result_np + soft_result_np_ud + soft_result_np_lr) / 3
        # soft_result_np = soft_result_np.transpose(2, 0, 1)
        # outputs = net(img)
        # print(outputs.shape)
        # outputs_soft = F.softmax(soft_result_np, dim=1)
        # result = soft_result_np.squeeze(dim=0)
        # result = result.argmax(dim=0).cpu().numpy()
    # mask = (result >= threshold).numpy()
        return soft_result_np
# def predict_softmax(net, img, device):
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)
#     with torch.no_grad():
#         outputs = net(img)
#         # print(outputs.shape)
#         outputs_soft = F.softmax(outputs, dim=1)
#     result = outputs_soft.squeeze(dim=0)
#     result = result.argmax(dim=0).cpu().numpy()
#     # mask = (result >= threshold).numpy()
#     return result


def predict_softmax_uint8(net, img, device):
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = net(img)
        # print(outputs.shape)
        outputs_soft = F.softmax(outputs, dim=1)
    result = outputs_soft.squeeze(dim=0)
    result = result.argmax(dim=0).cpu().numpy()
    result = result.astype(np.uint8)
    # mask = (result >= threshold).numpy()
    return result


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd

if __name__ == '__main__':
    args = get_args()
    is_compute_sdf = True
    snapshot_path = '../model/{}'.format(args.model)
    device = torch.device('cuda:2')

    net = UNet2(n_classes=1).to(device)
    save_mode_path = os.path.join(snapshot_path, 'best.pth')
    net.load_state_dict(torch.load(save_mode_path))
    net.eval()
    ##########process one img, PIL to tensor
    # args.img_path = '../../datasets/Glas/testC/KSBC_70.png'
    # img_path = '../../datasets/Glas/testC/'
    # img_testA_list = ['../../datasets/GlaS/testA_'+str(i+1)+'.bmp' for i range(60)]
    # img_testA_list = ['../../datasets/GlaS/testA_' + str(i + 1) + '.bmp' for i range(60)]
    # img_path = '../../datasets/GlaS/testA_1.bmp'
    # mask_path = '../../datasets/GlaS/testA_1_anno.bmp'
    img_path = '../../datasets/CRAG/test/images/test_10crop_3.png'
    mask_path = '../../datasets/CRAG/test/anno/test_10crop_3.png'
    # img_path = '../../datasets/Glas/testA/images/testA_43crop_3.png'
    # mask_path = '../../datasets/Glas/testA/anno/testA_43crop_3.png'
    img = Image.open(img_path)
    h, w = img.size
    # print(h, w)
    # img = img.resize((400, 400), Image.ANTIALIAS)

    # img
    gt_mask = Image.open(mask_path)
    gt_mask = np.array(gt_mask)
    original_img = np.array(img)
    cd_img = pre(original_img)
    norm = normalize_img(cd_img)
    results = inference_image(net, img, device)
    final_mask = postprocess(results, args.t)
    ##############
    # if args.model == 'UAMT':
    #     mask = predict_softmax_uint8(net, img, device)
    #     # mask0 = obj_mask(mask)
    # else:
    #     mask = predict(model, img, device, threshold=0.6)
    # labels = labeled = skimage.measure.label(mask)
    # eroded_mask = create_eroded_mask(labels)
#     results = np.zeros([2, 400, 400])
#     results[0] = eroded_mask
#     results[1] = labels
#     final_mask = postprocess(results)
# #################
    # result, input_image = inference_image(net, image)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

    ax1.imshow(original_img, interpolation='nearest')
    ax1.axis('off')
    ax1.set_title('Image')
    ax2.imshow(norm, interpolation='nearest')
    ax2.set_title('Prediction')
    ax2.axis('off')
    ax3.imshow(gt_mask, interpolation='nearest')
    ax3.set_title('GT mask')
    ax3.axis('off')
    fig.tight_layout()
    # plt.savefig()
    plt.show()

