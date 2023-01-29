import torch
import cv2
import skimage
from scipy import stats
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from node.metrics import Hausdorff
def Active_Contour_Loss(y_pred, y_true, device):
    """
    lenth term
    """
    # print(y_true)
    # device = torch.device('cuda:2')
    n, c, h, w = y_true.shape
    x = y_pred[:, 0, 1:, :] - y_pred[:, 0, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, 0, :, 1:] - y_pred[:, 0, :, :-1]

    delta_x = x[..., 1:, :-2] ** 2
    delta_y = y[..., :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """

    C_1 = torch.ones((h, w)).to(device)
    C_2 = torch.zeros((h, w)).to(device)

    region_in = torch.abs(torch.mean(y_pred[:, 0, ...] * ((y_true[:, 0, ...] - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - y_pred[:, 0, :, :]) * ((y_true[:, 0, ...] - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)

def ACWE_Loss(y_pred, y_true, device):
    """
    lenth term
    """
    # print(y_true)
    # device = torch.device('cuda:2')
    # _, h, w = y_true.shape
    x = y_pred[:, 0, 1:, :] - y_pred[:, 0, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, 0, :, 1:] - y_pred[:, 0, :, :-1]

    delta_x = x[..., 1:, :-2] ** 2
    delta_y = y[..., :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """
    Omega = y_pred > 0.5
    Omega = Omega.float()
    # print(y_pred.shape)
    N, C, H, W = y_pred.shape
    N1 = torch.zeros(N).to(device)
    N2 = torch.zeros(N).to(device)
     # exclude / 0 = + infty
    for i in range(N):
        N1[i] = torch.sum(Omega[i])
        N2[i] = H*W - N1[i]
    N1[N1 == 0] = 1
    N2[N2 == 0] = 1#### fixed some bugs
    # N1 = N1.view([N, 1, 1, 1])
    inside_inner = (y_true*Omega).reshape(N, C*H*W)
    C_1 = inside_inner.sum(dim=1)/N1
    outside_inner = (y_true*(1-Omega)).reshape(N, C*H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1, 1]).to(device)
    C_2 = C_2.view([N, 1, 1, 1]).to(device)


    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()

    region_in = torch.abs(torch.mean(Omega * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - Omega) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)

def ACWE_threshold_Loss(y_pred, y_true, device, threshold=0.5):
    """
    lenth term
    """
    # print(y_true)
    # device = torch.device('cuda:2')
    # _, h, w = y_true.shape
    x = y_pred[:, 0, 1:, :] - y_pred[:, 0, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, 0, :, 1:] - y_pred[:, 0, :, :-1]

    delta_x = x[..., 1:, :-2] ** 2
    delta_y = y[..., :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """
    Omega = y_pred > threshold
    Omega = Omega.float()
    # print(y_pred.shape)
    N, C, H, W = y_pred.shape
    N1 = torch.zeros(N).to(device)
    N2 = torch.zeros(N).to(device)
     # exclude / 0 = + infty
    for i in range(N):
        N1[i] = torch.sum(Omega[i])
        N2[i] = H*W - N1[i]
    N1[N1 == 0] = 1
    N2[N2 == 0] = 1#### fixed some bugs
    # N1 = N1.view([N, 1, 1, 1])
    inside_inner = (y_true*Omega).reshape(N, C*H*W)
    C_1 = inside_inner.sum(dim=1)/N1
    outside_inner = (y_true*(1-Omega)).reshape(N, C*H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1, 1]).to(device)
    C_2 = C_2.view([N, 1, 1, 1]).to(device)


    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()

    region_in = torch.abs(torch.mean(Omega * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - Omega) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)


def ACWE_threshold_Loss2(outputs, labels, device, threshold=0.5):
    """
    lenth term
    """
    y_true = labels[:, 1, ...]
    y_pred = outputs[:, 1, ...]
    # print(y_true)
    # device = torch.device('cuda:2')
    # _, h, w = y_true.shape
    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    delta_x = x[..., 1:, :-2] ** 2
    delta_y = y[..., :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """
    Omega = y_pred > threshold
    Omega = Omega.float()
    # print(y_pred.shape)
    N, H, W = y_pred.shape
    N1 = torch.zeros(N).to(device)
    N2 = torch.zeros(N).to(device)
     # exclude / 0 = + infty
    for i in range(N):
        N1[i] = torch.sum(Omega[i])
        N2[i] = H*W - N1[i]
    N1[N1 == 0] = 1
    N2[N2 == 0] = 1#### fixed some bugs
    # N1 = N1.view([N, 1, 1, 1])
    inside_inner = (y_true*Omega).reshape(N, H*W)
    C_1 = inside_inner.sum(dim=1)/N1
    outside_inner = (y_true*(1-Omega)).reshape(N, H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1]).to(device)
    C_2 = C_2.view([N, 1, 1]).to(device)


    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()

    region_in = torch.abs(torch.mean(Omega * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - Omega) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)

def ACWE_threshold_convex_Loss(outputs, labels, device, threshold=0.5):
    """
    lenth term
    """
    y_true = labels[:, 1, ...]
    y_pred = outputs[:, 1, ...]
    # print(y_true)
    # device = torch.device('cuda:2')
    # _, h, w = y_true.shape
    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    delta_x = x[..., 1:, :-2] ** 2
    delta_y = y[..., :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """
    Omega = y_pred > threshold
    Omega = Omega.float()
    # print(y_pred.shape)
    N, H, W = y_pred.shape
    N1 = torch.zeros(N).to(device)
    N2 = torch.zeros(N).to(device)
     # exclude / 0 = + infty
    for i in range(N):
        N1[i] = torch.sum(Omega[i])
        N2[i] = H*W - N1[i]
    N1[N1 == 0] = 1
    N2[N2 == 0] = 1#### fixed some bugs
    # N1 = N1.view([N, 1, 1, 1])
    inside_inner = (y_true*Omega).reshape(N, H*W)
    C_1 = inside_inner.sum(dim=1)/N1
    outside_inner = (y_true*(1-Omega)).reshape(N, H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1]).to(device)
    C_2 = C_2.view([N, 1, 1]).to(device)


    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()
    region_in = torch.abs(torch.mean(y_pred * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - y_pred) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)

def Per_batch_label_ACWE_Loss(outputs, labels, Si, Gi, device, threshold=0.5):
    """
    lenth term
    """
    # print(Si, Gi)
    Si = torch.from_numpy(Si).float().to(device)
    Gi = torch.from_numpy(Gi).float().to(device)
    y_true = labels*Gi
    y_pred = outputs*Si
    x = y_pred[1:, :] - y_pred[:-1, :]  # horizontal and vertical directions
    y = y_pred[:, 1:] - y_pred[:, :-1]

    delta_x = x[1:, :-2] ** 2
    delta_y = y[:-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)
    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """

    H, W = Si.shape
    N1 = torch.sum(Si)
    N2 = H*W - N1
    inside_inner = torch.sum(y_true*Si)
    outside_inner = torch.sum(y_true*(1-Si))
    C_1 = inside_inner/(N1+1)## avoid infty
    C_2 = outside_inner/(N2+1)

    region_in = torch.abs(torch.mean(y_pred * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - y_pred) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)

def obj_ACWE_threshold_Loss(outputs, labels, device, threshold=0.5):
    """
    lenth term
    """
    y_true = labels[:, 1, ...].cpu().numpy()
    Omega = (outputs[:, 1, ...]>threshold).cpu().numpy()
    N = Omega.shape[0]
    temp = 0
    for b in range(N):
        labels_b = skimage.measure.label(Omega[b])
        gt_labels_b = skimage.measure.label(y_true[b])
        listLabelS = np.unique(labels_b)
        listLabelG = np.unique(gt_labels_b)
        for label in listLabelS:
            Si = labels_b == label
            intersectlist = gt_labels_b[Si]
            if intersectlist.any():
                indexGi = stats.mode(intersectlist).mode
                Gi = (gt_labels_b == indexGi)
            else:
                tempDist = np.zeros((len(listLabelG), 1))
                for iLabelG in range(len(listLabelG)):
                    Gi = (gt_labels_b == listLabelG[iLabelG])
                    tempDist[iLabelG] = Hausdorff(Gi, Si)
                minIdx = np.argmin(tempDist)
                Gi = (gt_labels_b == listLabelG[minIdx])
            wi = np.sum(Si)/(np.sum(labels_b>0)+1)
            temp += wi*Per_batch_label_ACWE_Loss(outputs[b, 1, ...], labels[b, 1, ...], Si, Gi, device, threshold=0.5)
    return temp


def ACWE_Loss_dim2(y_pred, y_true, device):
    """
    lenth term
    """
    # print(y_true)
    # device = torch.device('cuda:2')
    # _, h, w = y_true.shape
    x = y_pred[1:, :] - y_pred[:-1, :]  # horizontal and vertical directions
    y = y_pred[:, 1:] - y_pred[:, :-1]

    delta_x = x[1:, :-2] ** 2
    delta_y = y[:-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """
    Omega = y_pred > 0.5
    Omega = Omega.float()
    # print(y_pred.shape)
    H, W = y_pred.shape
    N1 = torch.sum(Omega)
    N2 = H*W-N1
    inside_inner = y_true*Omega
    C_1 = torch.sum(inside_inner)/N1
    outside_inner = y_true*(1-Omega)
    C_2 = torch.sum(outside_inner)/N2
    region_in = torch.abs(torch.mean(Omega * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - Omega) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper
    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)

def _obj_dice(y_pred, y_true):
    Omega = y_pred > 0.5
    Omega = Omega.float()
    Omega_np = Omega.squeeze(dim=0).cpu().numpy()
    y_np = y_true.squeeze(dim=0).cpu().numpy()
    Omega_np = Omega_np.astype(np.uint8)
    y_np = y_np.astype(np.uint8)
    list1 = []
    list2 = []
    # print(Omega_np.shape)
    contours, hierarchy = cv2.findContours(Omega_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, h = cv2.findContours(y_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img2 = np.zeros_like(gt_mask)
    for i in range(len(contours)):
        # cimg1 = np.zeros_like(mask)
        img1 = np.zeros_like(Omega_np)
        cv2.drawContours(img1, contours, i, color=255, thickness=-1)
        c = img1 == 255
        # c = int(c)
        list1.append(c)
    for i in range(len(gt_contours)):
        # cimg1 = np.zeros_like(mask)
        img2 = np.zeros_like(y_np)
        cv2.drawContours(img2, gt_contours, i, color=255, thickness=-1)
        c = img2 == 255
        # c = int(c)
        list2.append(c)
    l1 = len(list1)
    l2 = len(list2)
    IoU = np.zeros([l2, l1])
    if l1 != 0 and l2 != 0:
        for i in range(l2):
            for j in range(l1):
                IoU[i, j] = dice(y_np*list2[i], Omega_np*list1[j])
        column_idx = np.argmax(IoU, 1)
        row_idx = np.argmax(IoU, 0)
        return list1, list2, row_idx, column_idx
    else:
        return [], [], 0, 0

def rm_obj_dice(y_pred, y_true):
    Omega = y_pred > 0.5
    Omega = Omega.float()
    Omega_np = Omega.squeeze(dim=0).cpu().numpy()
    y_np = y_true.squeeze(dim=0).cpu().numpy()
    Omega_np = Omega_np.astype(np.uint8)
    y_np = y_np.astype(np.uint8)
    list1 = []
    list2 = []
    # print(Omega_np.shape)
    contours, hierarchy = cv2.findContours(Omega_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, h = cv2.findContours(y_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img2 = np.zeros_like(gt_mask)
        # c = int(c)
    l1 = len(contours)
    l2 = len(gt_contours)

    if l1 != 0 and l2 != 0:
        i = np.random.choice(range(len(contours)))
        j = np.random.choice(range(len(gt_contours)))
        IoU1 = np.zeros([l1])
        IoU2 = np.zeros([l2])
        img1 = np.zeros_like(Omega_np)
        cv2.drawContours(img1, contours, i, color=255, thickness=-1)
        random_p = img1 == 255
        img2 = np.zeros_like(y_np)
        cv2.drawContours(img2, gt_contours, j, color=255, thickness=-1)
        random_gt = img2 == 255
        for idx in range(l2):
            img2 = np.zeros_like(y_np)
            cv2.drawContours(img2, gt_contours, idx, color=255, thickness=-1)
            IoU2[idx] = dice(y_np*(img2==255), Omega_np*random_p)
        for idx1 in range(l1):
            img1 = np.zeros_like(Omega_np)
            cv2.drawContours(img1, contours, idx1, color=255, thickness=-1)
            IoU1[idx1] = dice(y_np * random_gt, Omega_np * (img1==255))
        gt_idx = np.argmax(IoU2)
        img_2 = np.zeros_like(y_np)
        cv2.drawContours(img_2, gt_contours, gt_idx, color=255, thickness=-1)
        gt_maxiou = img_2 == 255
        p_idx  = np.argmax(IoU1)
        img_1 = np.zeros_like(Omega_np)
        cv2.drawContours(img_1, contours, p_idx, color=255, thickness=-1)
        p_maxiou = img_1 == 255
        return random_gt, p_maxiou, random_p, gt_maxiou
    else:
        return [], [], [], []

def rm_obj_dice_loss_batch(y_pred, y_true, device):
    y_true = y_true.squeeze(dim=0)
    random_gt, p_maxiou, random_p, gt_maxiou = rm_obj_dice(y_pred, y_true)
    if len(random_gt) > 1 and len(random_gt) > 1:
        random_gt = torch.from_numpy(random_gt).float().to(device)
        p_maxiou  = torch.from_numpy(p_maxiou).float().to(device)
        random_p = torch.from_numpy(random_p).float().to(device)
        gt_maxiou = torch.from_numpy(gt_maxiou).float().to(device)
        return dice_loss(y_pred*random_p, y_true*gt_maxiou)\
               + dice_loss(y_pred*p_maxiou, y_true*random_gt)\
               + dice_loss(y_pred, y_true)
    else:
        return dice_loss(y_pred, y_true)

def rm_obj_ACWE_loss_batch(y_pred, y_true, device):
    y_pred = y_pred.squeeze(dim=0)
    y_true = y_true.squeeze(dim=0)
    random_gt, p_maxiou, random_p, gt_maxiou = rm_obj_dice(y_pred, y_true)
    if len(random_gt) >= 1 and len(random_gt) >= 1:
        random_gt = torch.from_numpy(random_gt).float().to(device)
        p_maxiou  = torch.from_numpy(p_maxiou).float().to(device)
        random_p = torch.from_numpy(random_p).float().to(device)
        gt_maxiou = torch.from_numpy(gt_maxiou).float().to(device)
        return ACWE_Loss_dim2(y_pred*random_p, y_true*gt_maxiou, device)\
               + ACWE_Loss_dim2(y_pred*p_maxiou, y_true*random_gt, device)
    else:
        return 0

def obj_dice_loss_batch(y_pred, y_true, device):
    Omega = y_pred > 0.5
    Omega = Omega.float()
    Omega = Omega.squeeze(dim=0)
    y_true = y_true.squeeze(dim=0)
    list1, list2, row_idx, column_idx = _obj_dice(y_pred, y_true)
    if len(list1) >= 1 and len(list2) >= 1:
        s1 = torch.sum(Omega).float().to(device)
        s2 = torch.sum(y_true).float().to(device)
        print(s1, s2)
        sum1 = torch.tensor(0.0).to(device)
        sum2 = torch.tensor(0.0).to(device)
        for i in range(len(list1)):
            idx = torch.from_numpy(list1[i]).float().to(device)
            idx2 = torch.from_numpy(list2[row_idx[i]]).float().to(device)
            # if isinstance(s1, )
            sum1 += torch.sum(idx).to(device)/s1*dice_loss(idx*y_pred, y_true*idx2)
        for i in range(len(list2)):
            gt_idx = torch.from_numpy(list2[i]).float().to(device)
            idx1 = torch.from_numpy(list1[column_idx[i]]).float().to(device)
            sum2 += torch.sum(gt_idx).to(device)/s2*dice_loss(y_true*gt_idx, y_true*idx1)
        return sum1+sum2
    else:
        return 0

def obj_dice_loss(y_pred, y_true, device):
    N, C, H, W = y_pred.shape
    sum = torch.tensor(0.0).to(device)
    for n in range(N):
        sum += obj_dice_loss_batch(y_pred[n], y_true[n], device)
    return sum/N

def rm_obj_dice_loss(y_pred, y_true, device):
    # print(y_pred.shape)
    N, C, H, W = y_pred.shape
    sum = torch.tensor(0.0).to(device)
    for n in range(N):
        sum += rm_obj_dice_loss_batch(y_pred[n], y_true[n], device)
    return sum/N

def rm_obj_ACWE_loss(y_pred, y_true, device):
    # print(y_pred.shape)
    N, C, H, W = y_pred.shape
    sum = torch.tensor(0.0).to(device)
    for n in range(N):
        sum += rm_obj_ACWE_loss_batch(y_pred[n], y_true[n], device)
    return sum/N

def dice(score, target):
    target = np.float32(target)
    smooth = 1e-5
    intersect = np.sum(score * target)
    y_sum = np.sum(target * target)
    z_sum = np.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return dice

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss



def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)


    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
