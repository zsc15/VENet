import torch
import cv2
import skimage
from scipy import stats
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from node.metrics import Hausdorff

# y_true = labels[:, dim, ...]
# y_pred = outputs[:, dim, ...]
def Active_Contour_Loss(y_pred, y_true, device):
    """
    lenth term
    """
    # print(y_true)
    # device = torch.device('cuda:2')
    _, h, w = y_true.shape
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

def Active_Contour_Loss_dim(y_pred, y_true, device, dim=0):
    """
    lenth term
    """
    y_true = y_true[:, dim, ...]
    y_pred = y_pred[:, dim, ...]
    # print(y_true)
    # device = torch.device('cuda:2')
    _, h, w = y_true.shape
    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    delta_x = x[..., 1:, :-2] ** 2
    delta_y = y[..., :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

    """
    region term
    """

    C_1 = torch.ones((h, w)).to(device)
    C_2 = torch.zeros((h, w)).to(device)

    region_in = torch.abs(torch.mean(y_pred * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - y_pred) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

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

def ACWE_threshold_Loss3(y_pred, y_true, device, threshold=0.5):
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

def TV_length(outputs):
    y_pred = outputs[:, 0, ...]
    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]
    delta_x = x[..., 1:, :-2] ** 2
    delta_y = y[..., :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)
    length = torch.mean(torch.sqrt(delta_u + 0.00000001))
    return length

def ACWE_threshold_convex_with_contour_Loss(outputs, labels, contours, device, dim=1, threshold=0.5):
    """
    lenth term
    """
    y_true = labels[:, dim, ...]
    y_pred = outputs[:, dim, ...]
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
    Omega = (y_pred > threshold).float() * (contours[:, 0, ...] < 0.5).float()
    Omega_out = (y_pred <= threshold).float() * (contours[:, 0, ...] < 0.5).float()
    # Omega = Omega_in.float()
    # Omega_out = Omega_out.float()
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
    outside_inner = (y_true*Omega_out).reshape(N, H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1]).cuda(device.index)
    C_2 = C_2.view([N, 1, 1]).cuda(device.index)
    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()
    region_in = torch.abs(torch.mean(y_pred * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - y_pred) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out)

def ACWE_threshold_convex_Loss(outputs, labels,  device, dim=1, threshold=0.5):
    """
    lenth term
    """
    y_true = labels[:, dim, ...]
    y_pred = outputs[:, dim, ...]
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
    Omega = (y_pred > threshold).float()
    Omega_out = (y_pred <= threshold).float()
    # Omega = Omega_in.float()
    # Omega_out = Omega_out.float()
    # print(y_pred.shape)
    N, H, W = y_pred.shape
    N1 = torch.zeros(N).cuda(device.index)
    N2 = torch.zeros(N).cuda(device.index)
     # exclude / 0 = + infty
    for i in range(N):
        N1[i] = torch.sum(Omega[i])
        N2[i] = H*W - N1[i]
    N1[N1 == 0] = 1
    N2[N2 == 0] = 1#### fixed some bugs
    # N1 = N1.view([N, 1, 1, 1])
    inside_inner = (y_true*Omega).reshape(N, H*W)
    C_1 = inside_inner.sum(dim=1)/N1
    outside_inner = (y_true*Omega_out).reshape(N, H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1]).cuda(device.index)
    C_2 = C_2.view([N, 1, 1]).cuda(device.index)
    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()
    region_in = torch.abs(torch.mean(y_pred * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - y_pred) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.

    return lenth + lambdaP * (mu * region_in + region_out), torch.mean(C_1), torch.mean(C_2)

def ACWE_threshold_Loss(outputs, labels,  device, dim=1, threshold=0.5):
    """
    lenth term
    """
    y_true = labels[:, dim, ...]
    y_pred = outputs[:, dim, ...]
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
    Omega = (y_pred > threshold).float()
    Omega_out = (y_pred <= threshold).float()
    # Omega = Omega_in.float()
    # Omega_out = Omega_out.float()
    # print(y_pred.shape)
    N, H, W = y_pred.shape
    N1 = torch.zeros(N).cuda(device.index)
    N2 = torch.zeros(N).cuda(device.index)
     # exclude / 0 = + infty
    for i in range(N):
        N1[i] = torch.sum(Omega[i])
        N2[i] = H*W - N1[i]
    N1[N1 == 0] = 1
    N2[N2 == 0] = 1#### fixed some bugs
    # N1 = N1.view([N, 1, 1, 1])
    inside_inner = (y_true*Omega).reshape(N, H*W)
    C_1 = inside_inner.sum(dim=1)/N1
    outside_inner = (y_true*Omega_out).reshape(N, H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1]).cuda(device.index)
    C_2 = C_2.view([N, 1, 1]).cuda(device.index)
    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()
    region_in = torch.abs(torch.mean(Omega * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out = torch.abs(torch.mean((1 - Omega) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.
    mu = 1  # mu parameter could be various.
    return lenth + lambdaP * (mu * region_in + region_out), torch.mean(C_1), torch.mean(C_2)

def generalized_variational_Loss(outputs, labels, device, args, dim=1, threshold=0.5):
    """
    lenth term
    """
    y_true = labels[:, dim, ...]
    y_pred = outputs[:, dim, ...]
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
    Omega = (y_pred > threshold).float()
    Omega_out = (y_pred <= threshold).float()
    # Omega = Omega_in.float()
    # Omega_out = Omega_out.float()
    # print(y_pred.shape)
    N, H, W = y_pred.shape
    N1 = torch.zeros(N).cuda(device.index)
    N2 = torch.zeros(N).cuda(device.index)
     # exclude / 0 = + infty
    for i in range(N):
        N1[i] = torch.sum(Omega[i])
        N2[i] = H*W - N1[i]
    N1[N1 == 0] = 1
    N2[N2 == 0] = 1#### fixed some bugs
    # N1 = N1.view([N, 1, 1, 1])
    inside_inner = (y_true*Omega).reshape(N, H*W)
    C_1 = inside_inner.sum(dim=1)/N1
    outside_inner = (y_true*Omega_out).reshape(N, H*W)
    C_2 = outside_inner.sum(dim=1)/N2
    C_1 = C_1.view([N, 1, 1]).cuda(device.index)
    C_2 = C_2.view([N, 1, 1]).cuda(device.index)
    # C_1 = torch.ones((h, w)).cuda()
    # C_2 = torch.zeros((h, w)).cuda()
    region_in0 = torch.abs(torch.mean(y_pred * ((y_true - C_1) ** 2)))  # equ.(12) in the paper
    region_out0 = torch.abs(torch.mean((1 - y_pred) * ((y_true - C_2) ** 2)))  # equ.(12) in the paper
    region_in = torch.abs(torch.mean(Omega * ((y_true - C_1) ** 2)))
    region_out = torch.abs(torch.mean((1 - Omega) * ((y_true - C_2) ** 2)))
    return lenth + args.mu * (region_in + region_out) + args.nu * (region_in0 + region_out0), torch.mean(C_1), torch.mean(C_2)

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss










