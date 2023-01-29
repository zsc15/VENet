import numpy as np
from sklearn.cluster import KMeans
import createsample3 as cs
import TensorflowUtils as utils
import openslide
import multiprocessing
import cv2
from PIL import Image
import copy
from skimage import morphology, io
import scipy.ndimage as ndi
import time
import imageio
t0 = time.time()
logs_dir = './model/5.24'
size1 = 400
size2 = 400
resize1 = 400
resize2 = 400
batchsize = 4
minsize = 50 * 50
suffix = '1602532'
xml_file = '16279801.xml'
source = openslide.open_slide('16279801.svs')
[w, h] = source.level_dimensions[0]
patchpred = []
patch1 = []
t1 = time.time()
dim = source.dimensions
countlevel = source.level_count
dim1 = source.level_dimensions[countlevel - 1]
down = source.level_downsamples[countlevel - 1]
pad = np.int(100 / down)
shift = np.int(200 / down)
reimagesize = np.int(400 / down)
def segmentblock(kclus1, block, down):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(kclus1.astype('uint8'))
    maxnum = np.max(labels)
    if maxnum > 0:
        for i in range(1, maxnum + 1):
            if stats[i, 4] > np.int((size1/down)) ** 2:
                height_min = stats[i, 1]
                height_max = stats[i, 1] + stats[i, 3]
                width_min = stats[i, 0]
                width_max = stats[i, 0] + stats[i, 2]
                if (width_max - width_min + 1) > np.int((size1/down)) and (height_max - height_min + 1) > np.int((size1/down)):
                    if ((width_max - width_min) + 1) * ((height_max - height_min) + 1) < np.sum(kclus1[height_min:height_max, width_min:width_max]) * 2:
                        info = [height_min, height_max, width_min, width_max]
                        block.append(info)
                    else:
                        pp = np.zeros_like(labels, dtype='int')
                        pp[height_min: np.int(height_min + (height_max - height_min) / 2),
                        width_min: np.int(width_min + (width_max - width_min) / 2)] = 1
                        pp1 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp1, block, down))


                        pp = np.zeros_like(labels, dtype='int')
                        pp[np.int(height_min + (height_max - height_min) / 2): height_max,
                                  width_min: np.int(width_min + (width_max - width_min) / 2)] = 1
                        pp2 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp2, block, down))


                        pp = np.zeros_like(labels, dtype='int')
                        pp[height_min: np.int(height_min + (height_max - height_min) / 2),
                        np.int(width_min + (width_max - width_min) / 2): width_max] = 1
                        pp3 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp3, block, down))


                        pp = np.zeros_like(labels, dtype='int')
                        pp[np.int(height_min + (height_max - height_min) / 2):height_max,
                                  np.int(width_min + (width_max - width_min) / 2):width_max] = 1
                        pp4 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp4, block, down))
    return block
patch = np.array(source.read_region((0, 0), countlevel - 1, (dim1[0], dim1[1])))
patch = patch[:, :, 0:3]
ac = copy.deepcopy(patch)
ac1 = np.reshape(ac, [dim1[0] * dim1[1], 3], 'F')
clus = copy.deepcopy(ac1)
mod = KMeans(n_clusters=2, n_init = 1)
kclus = mod.fit_predict(clus)
kclus1 = np.reshape(kclus, [dim1[1], dim1[0]], 'F')
pan1 = np.sum((ac[:, :, 0] + ac[:, :, 1] + ac[:, :, 2]) * kclus1) / (np.sum(kclus1))
pan2 = np.sum((ac[:, :, 0] + ac[:, :, 1] + ac[:, :, 2]) * (1 - kclus1)) / (dim1[0] * dim1[1] - np.sum(1 - kclus1))
if pan2 < pan1:
    kclus1 = copy.deepcopy(1 - kclus1)
kclus1p = copy.deepcopy(kclus1)
bool1 = kclus1p != 0
bool2 = morphology.remove_small_objects(bool1, min_size=np.int(250**2/(down**2)), connectivity=2)
kclus1 = kclus1 * 0
kclus1[bool2] = 1
kclus1 = ndi.binary_fill_holes(kclus1)
kclus1 = np.array(kclus1, 'uint8')
# imageio.imwrite('E:/sa/'+suffix+'masks'+'.png', kclus1*255)
def readimage(picloc):
    if np.sum(kclus1[picloc[0]:picloc[0]+reimagesize, picloc[1]:picloc[1]+reimagesize]) >= np.int(reimagesize**2*3/4):
        region = np .array(source.read_region((np.int(picloc[1]*down), np.int(picloc[0]*down)), 0, (size1, size1)))
        region = copy.deepcopy(region[:, :, 0:3])
        if (picloc[0]-picloc[2]) % (4*pad) == 0 and (picloc[1]-picloc[3]) % (4*pad) == 0:
            preimage = pre(region)
            return [region, [np.int(picloc[0]*down), np.int(picloc[1]*down)], preimage, [picloc[0], picloc[1]]]
        else:
            return [region, [np.int(picloc[0]*down), np.int(picloc[1]*down)],  0, 0]
    else:
        return 0


def postprocess(predseg1, pred1):
    pred1[(pred1 == 0) & (predseg1 != 0)] = copy.deepcopy(predseg1[(pred1 == 0) & (predseg1 != 0)])
    pp = copy.deepcopy(pred1)
    bool1 = pp == 1
    bool2 = morphology.remove_small_objects(bool1, min_size=minsize, connectivity=2)
    pp1 = np.zeros_like(pp, dtype=np.float64)
    pp1[bool2] = 1
    pp1 = ndi.binary_fill_holes(pp1).astype(int)
    return pp1


def pre(input1):
    input12 = utils.preprocess(input1)
    return input12

def pre1(loc1, loc2):
    region = np.array(source.read_region((np.int(loc2), np.int(loc1)), 0, (size1, size2)))
    input1 = copy.deepcopy(region[:, :, 0:3])
    normalize = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    input1 = Image.fromarray(input1.astype('uint8')).convert('RGB')
    #input12 = utils.preprocess(input1)
    input12 = np.array(
        utils.resize_img(
            img_pil=utils.crop_img(input1, 1),
            scale=1,
            type='image',
            resize_size=[resize1, resize2]
        ),
        dtype=np.uint8
    )
    input12 = utils.trans(input12, normalize)
    input12 = input12.transpose(1, 2, 0)
    return [input12, loc1, loc2]


if __name__ == "__main__":
    ####在大图中识别重点区域并分块
    kclus2 = copy.deepcopy(kclus1)
    block = []
    blocks = segmentblock(kclus2, block, down)
    results = []
    picloc = []
    pic = []
    for block in blocks:
        height_min = np.max([block[0] - shift, 0])
        height_max = np.min([block[1] + shift, dim1[1]])
        width_min = np.max([block[2] - shift, 0])
        width_max = np.min([block[3] + shift, dim1[0]])
        s1 = np.int((height_max - height_min - reimagesize) / pad)
        s2 = np.int((width_max - width_min - reimagesize) / pad)
        for i in range(s1):
            for j in range(s2):
                picloc.append([i * pad + height_min, j * pad + width_min, height_min, width_min])
    pool = multiprocessing.Pool(processes=6)
    for num in range(len(picloc)):
        pic.append(pool.apply_async(readimage, (picloc[num],)))
    pool.close()
    pool.join()
    whole = []
    figure = []
    preimage = []
    preloc = []
    for num in range(len(pic)):
        whole.append(pic[num].get())
    for num in range(len(whole)):
        if isinstance(whole[num], int) == False:
            if isinstance(whole[num][2], int) == False:
                figure.append(whole[num][0])
                preimage.append(whole[num][2])
                preloc.append(whole[num][1])
    pred = np.zeros([dim[1], dim[0]], 'float32')
    # for i in range(len(figure)):
    #     imageio.imwrite('E:/sa/'+str(i)+'.png', figure[i])
    t2 = time.time()
    print(t2-t1)
    i = 0
    device = 'cuda:0'
    print(np.shape(preimage))

    pred, loc2 = cs.samplemake(preimage, preloc, w, h, logs_dir, pred, batchsize, minsize)
    #pred, loc2 = cs.samplemake(preimage, preloc, w, h, logs_dir, pred, batchsize, minsize, device)
    print(loc2)
    pool = multiprocessing.Pool(processes=6)  # 创建12个进程
    input1 = []
    for i in range(len(loc2)):
        input1.append(pool.apply_async(pre, (loc2[i][0], loc2[i][1], )))
    input2 = []
    loc2 = []
    count = len(input1)
    for num in range(count):
        input2.append(input1[num].get())
    input3 = []
    for ele in input2:
        input3.append(ele[0])
    for ele in input2:
        loc2.append(ele[1:])
    pool.close()
    pool.join()
    del input2
    if len(input3) != 0:
        patch1 = cs.blockremove(len(loc2), batchsize, size1, size2, logs_dir, pic, input3)
        #patch1 = cs.blockremove(len(loc2), batchsize, resize1, resize2, logs_dir, pic, input3, device)
        del input3
        del pic
        results2 = []
        pool2 = multiprocessing.Pool(processes=6)
        for i in range(len(loc2)):
            loc3 = loc2[i]
            patchpred.append(cv2.resize(pred[np.int(loc3[0]):np.int(loc3[0]) + size2,
                             np.int(loc3[1]):np.int(loc3[1]) + size1].astype(np.uint8), (resize1, resize2)))
        for i in range(len(loc2)):
            results2.append(pool2.apply_async(postprocess, (patch1[i, :, :], patchpred[i],)))
        pool2.close()
        pool2.join()
        for i in range(len(loc2)):
            loc3 = loc2[i]
            pred[np.int(loc3[0]):np.int(loc3[0]) + size2,
            np.int(loc3[1]):np.int(loc3[1]) + size1] = cv2.resize(results2[i].get().astype(np.uint8), (size1, size2))
        pred = ndi.binary_fill_holes(pred)
        pred = np.array(pred, 'uint8')
        cs.writexml(xml_file, pred)
    else:
        print('The WSI has no obvious high grade intraepithelial neoplasia')