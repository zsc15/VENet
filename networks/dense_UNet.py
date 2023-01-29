import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
# from torchsummary import summary
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from medpy.io import load
from skimage.transform import resize
from multiprocessing.dummy import Pool as ThreadPool

# pytorch implementation of HDenseUNet
# the base file is from the repo https://github.com/thangylvp/HDenseUet/blob/master/HDenseUnet.py
# the repo contains encoer decoder format instead of UNet

device = 'cuda'


class Scale(nn.Module):
    def __init__(self, num_feature):
        super(Scale, self).__init__()
        self.num_feature = num_feature
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True)

    def forward(self, x):
        y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.num_feature):
            y[:, i, :, :] = x[:, i, :, :].clone() * self.gamma[i] + self.beta[i]
        return y


class dense_block(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0, weight_decay=1e-4, grow_nb_filters=True):
        super(dense_block, self).__init__()
        for i in range(nb_layers):
            layer = conv_block(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        # are the module names dictionary sorted?
        # else how do we process denseLayer1 before denseLayer2
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


# some problem with the transition class, def forward absent?
class _Transition(nn.Sequential):
    def __init__(self, num_input, num_output, drop=0):
        super(_Transition, self).__init__()

        self.drop = drop
        self.norm = nn.BatchNorm2d(num_input)
        self.scale = Scale(num_input)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input, num_output, (1, 1), bias=False)
        weight_init(self)
        # self.pool = nn.AvgPool2d(kernel_size= 2, stride= 2)

    # added this part as it was not present in the original code
    def forward(self, x):
        out = self.conv(self.relu(self.scale(self.norm(x))))
        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)
        return F.avg_pool2d(out, 2)


class conv_block(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0, weight_decay=1e-4):
        super(conv_block, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm2d(nb_inp_fea, eps=eps, momentum=1))
        self.add_module('scale1', Scale(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2d1', nn.Conv2d(nb_inp_fea, 4 * growth_rate, (1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm2d(4 * growth_rate, eps=eps, momentum=1))
        self.add_module('scale2', Scale(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2d2', nn.Conv2d(4 * growth_rate, growth_rate, (3, 3), padding=(1, 1), bias=False))

        weight_init(self)

    def forward(self, x):
        out = self.conv2d1(self.relu1(self.scale1(self.norm1(x))))
        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        out = self.conv2d2(self.relu2(self.scale2(self.norm2(out))))
        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        return out


def weight_init(net):
    for name, m in net.named_children():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


class denseUnet(nn.Module):
    def __init__(self, in_c=1, out_c=1, growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, drop_rate=0,
                 weight_decay=1e-4, num_classes=1000, reduction=0.0):
        super(denseUnet, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5
        compression = 1 - reduction
        # initial convolution
        # self.conv0_ = nn.Conv2d(3, nb_filter, kernel_size=7, stride=2,
        #                        padding=3, bias=False)
        self.conv0_ = nn.Conv2d(in_c, nb_filter, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.norm0_ = nn.BatchNorm2d(nb_filter, eps=eps)
        self.scale0_ = Scale(nb_filter)
        self.ac0_ = nn.ReLU(inplace=True)
        self.pool0_ = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # dense block followed by transition
        num_layer = block_config
        self.block1 = dense_block(num_layer[0], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[0] * growth_rate
        self.trans1 = _Transition(nb_filter, math.floor(nb_filter * compression))
        nb_filter = int(nb_filter * compression)

        self.block2 = dense_block(num_layer[1], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[1] * growth_rate
        self.trans2 = _Transition(nb_filter, math.floor(nb_filter * compression))
        nb_filter = int(nb_filter * compression)

        self.block3 = dense_block(num_layer[2], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[2] * growth_rate
        # to store the filter from the 3rd denso block as this will determine the
        # number of input channels for self.conv
        nb_filter3 = nb_filter
        self.trans3 = _Transition(nb_filter, math.floor(nb_filter * compression))
        nb_filter = int(nb_filter * compression)

        self.block4 = dense_block(num_layer[3], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[3] * growth_rate
        self.bn2_ = nn.BatchNorm2d(nb_filter, eps=eps, momentum=1)
        self.scale2_ = Scale(nb_filter)
        self.ac2_ = nn.ReLU(inplace=True)

        print('nb_filter: ', nb_filter)

        # the other half of the UNet
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(nb_filter3, 2208, kernel_size=1, padding=0)

        self.conv0 = nn.Conv2d(2 * 2208, 768, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(768, momentum=1)
        self.ac0 = nn.ReLU(inplace=True)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(2 * 768, 384, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(384, momentum=1)
        self.ac1 = nn.ReLU(inplace=True)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(2 * 384, 96, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(96, momentum=1)
        self.ac2 = nn.ReLU(inplace=True)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(2 * 96, 96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(96, momentum=1)
        self.ac3 = nn.ReLU(inplace=True)

        self.up4 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm2d(64, momentum=1)
        self.ac4 = nn.ReLU(inplace=True)

        # last convolution
        # self.conv5 = nn.Conv2d(64, 3, kernel_size=1)
        # the output is a segmentation map consisting of two channels(= #classes)
        self.conv5 = nn.Conv2d(64, out_c, kernel_size=1)
        self.conv52 = nn.Conv2d(64, 2, kernel_size=1)
        self.tanh = nn.Tanh()

        # init_weights(self)
        weight_init(self)

    # this part is not UNet, this is encoder decoder
    def forward(self, x):
        box = []
        out = self.ac0_(self.scale0_(self.norm0_(self.conv0_(x))))
        box.append(out)
        print('box[0] size: ', box[0].size())
        out = self.pool0_(out)

        out = self.block1(out)
        box.append(out)
        print('box[1] size: ', box[1].size())
        out = self.trans1(out)

        out = self.block2(out)
        box.append(out)
        print('box[2] size: ', box[2].size())
        out = self.trans2(out)

        out = self.block3(out)
        box.append(out)
        print('box[3] size: ', box[3].size())
        out = self.trans3(out)

        out = self.block4(out)

        out = self.ac2_(self.scale2_(self.bn2_(out)))
        box.append(out)
        print('box[4] size: ', box[4].size())

        up0 = self.up(out)
        line0 = self.conv(box[3])
        # up0_sum = add([line0, up0])
        up0_sum = torch.cat((line0, up0), dim=1)
        out = self.ac0(self.bn0(self.conv0(up0_sum)))

        up1 = self.up1(out)
        # up1_sum = add([box[2], up1])
        up1_sum = torch.cat((box[2], up1), dim=1)
        out = self.ac1(self.bn1(self.conv1(up1_sum)))

        up2 = self.up2(out)
        # up2_sum = add([box[1], up2])
        up2_sum = torch.cat((box[1], up2), dim=1)
        out = self.ac2(self.bn2(self.conv2(up2_sum)))

        up3 = self.up3(out)
        # up3_sum = add([box[0], up3])
        up3_sum = torch.cat((box[0], up3), dim=1)
        out = self.ac3(self.bn3(self.conv3(up3_sum)))

        up4 = self.up4(out)
        out = self.ac4(self.bn4(self.dropout(self.conv4(up4))))
        segout = self.conv52(out)
        out = self.conv5(out)

        return self.tanh(out), segout