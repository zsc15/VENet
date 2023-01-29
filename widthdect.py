# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pylab as plt
import PIL
from Utils import SeparateStains
from Utils import widthdetect
import scipy.misc as misc
import cv2 as cv
import copy
from scipy import signal
from sklearn.cluster import KMeans
import math
from sklearn.metrics import silhouette_score
import scipy.misc as misc
import matplotlib.pylab as plt
def measure(imageRGB, b1):
    # cv.setNumThreads(0)
    imagesize = np.shape(imageRGB)
    b1r = np.reshape(b1, [imagesize[0] * imagesize[1], 1], 'F')
    imagegroup = np.zeros([imagesize[0] * imagesize[1], 3], dtype='uint8')
    for i in range(imagesize[2]):
        imagegroup[:, i] = np.reshape(imageRGB[:, :, i], [imagesize[0] * imagesize[1]], 'F')
    imagecluster = []
    for i in range(imagesize[0] * imagesize[1]):
        if b1r[i] != 0:
            imagecluster.append(imagegroup[i, :])
    mod = KMeans(n_clusters=3)
    imageresult = mod.fit_predict(imagecluster)
    t1 = np.zeros([3])
    t2 = np.zeros([3])
    t3 = np.zeros([3])
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(imagecluster)):
        if imageresult[i] == 0:
            t1 = t1 + imagecluster[i]
            count1 += 1
        if imageresult[i] == 1:
            t2 = t2 + imagecluster[i]
            count2 += 1
        if imageresult[i] == 2:
            t3 = t3 + imagecluster[i]
            count3 += 1
    t1 = np.mean(t1 / count1)
    t2 = np.mean(t2 / count2)
    t3 = np.mean(t3 / count3)
    if t1 > 180 or t2 > 180 or t3 > 180:
        bnum = 2
    else:
        bnum = 1
    #bnum = 2
    ele = 1
    # imageURL = 'D:/redo3/images/training/1972.png'
    # b1 = misc.imread('D:/redo3/annotations/training/1972.png')
    # imageRGB = misc.imread(imageURL)
    _, b2 = cv.connectedComponents(b1)
    bound1 = cv.Canny(b1, 0, 1)
    bound1[bound1 != 0] = 1
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
    imageHDAB = SeparateStains(imageRGB, RGBtoHDAB)
    ac = copy.deepcopy(imageHDAB[:, :, 0])
    bc = copy.deepcopy(imageHDAB[:, :, 1])
    b3 = copy.deepcopy(b2)
    b3[b3 != ele] = 0
    b3[b3 != 0] = 1
    b3 = np.uint8(b3)
    imagesize = b3.shape
    edge1 = cv.Canny(b3, 0, 1)
    area = np.sum(b3)
    ac = b3 * ac
    bc = b3 * bc
    b31 = np.reshape(b3, [imagesize[0] * imagesize[1]], 'F')
    ac1 = np.reshape(ac, [imagesize[0] * imagesize[1]], 'F')
    bc1 = np.reshape(bc, [imagesize[0] * imagesize[1]], 'F')
    clus = ac1[b31 == 1]
    anchorloc = np.where(clus == np.min(clus))
    b31 = np.reshape(b3, [imagesize[0] * imagesize[1]], 'F')
    locclus = np.where(b31 == 1)
    mod = KMeans(n_clusters=2)
    cclus = [[i] for i in clus]
    kclus = mod.fit_predict(cclus)
    ac1[locclus[0][kclus == kclus[anchorloc[0][0]]]] = 0
    ac = np.reshape(ac1, [imagesize[0], imagesize[1]], 'F')
    ##calculate cytoplasm
    if bnum == 2:
        clus = bc1[b31 == 1]
        anchorloc = np.where(clus == np.min(clus))
        b31 = np.reshape(b3, [imagesize[0] * imagesize[1]], 'F')
        locclus = np.where(b31 == 1)
        mod = KMeans(n_clusters=bnum)
        cclus = [[i] for i in clus]
        kclus = mod.fit_predict(cclus)
        bc1[locclus[0][kclus == kclus[anchorloc[0][0]]]] = 0
        bc = np.reshape(bc1, [imagesize[0], imagesize[1]], 'F')
        bc[bc != 0] = 1
    else:
        bc[bc != 0] = 1
    ##calculate nucleu
    # ac1 = cv.GaussianBlur(ac, (11, 11), 1)
    # ac2 = signal.order_filter(ac1, np.ones([11, 11]), 11 * 11 - 1)
    # judge1 = (ac1 == ac2) & (ac1 > 0)
    # [w, h] = np.shape(judge1)
    # di = np.zeros([w, h])
    # di[judge1] = 255
    # [x, y] = np.where(di == 255)
    ac[ac != 0] = 1
    edge1[edge1 != 0] = 1
    perimeter = np.sum(edge1)
    # calculate nucleu width
    [width, LOC] = widthdetect(ac, b3, edge1)
    # caluculate ratio
    if np.max(np.sum(bc) - np.sum(ac), 0) != 0:
        ratio = np.sum(ac) / (np.sum(bc) - np.sum(ac))
    else:
        ratio = 10 ** 12
    # calculate color
    colorchannel = np.zeros([np.shape(imageRGB)[-1]])
    for i in range(np.shape(imageRGB)[-1]):
        colorchannel[i] = np.sum(imageRGB[:, :, i] * bc) / np.sum(bc)

    # edge11 = np.reshape(edge1, [imagesize[0]*imagesize[1]], 'F')
    # edgeloc = np.where(edge11==1)
    # edgeloc = edgeloc[0]
    # ac1 = np.reshape(ac, [imagesize[0]*imagesize[1]],'F')
    # for loc in LOC:
    #     ac1[np.int(loc)] = 3
    # ac1[edgeloc] = 2
    # ac2 = np.reshape(ac1, [imagesize[0],imagesize[1]], 'F')
    # plt.figure()
    # plt.imshow(ac2, cmap = 'Greys_r')
    # sigma = np.std(width)
    # ave = np.mean(width)
    # width1 = width - ave
    # loc11 = np.where(np.abs(width1) > 3*sigma)
    # for locnum in loc11[0]:
    #     width[locnum] = 0
    # if (len(width)-len(loc11[0])) != 0:
    #     avewidth = sum(width)/(len(width)-len(loc11[0]))
    # else:
    #     avewidth = 0
    width1 = np.sort(width)
    width1 = width1[width1 != 0]
    width2 = copy.deepcopy(width1[round(len(width1) * 0.65):round(len(width1) * 0.95)])
    if len(width2) != 0:
        avewidth = sum(width2) / (len(width2))
    else:
        avewidth = 0
    x = [1]
    y = [1]
    return avewidth, ratio, colorchannel[0], colorchannel[1], colorchannel[2], np.float64(
        perimeter ** 2 / (4 * math.pi * area)), x, y