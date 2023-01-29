from __future__ import print_function
# import tensorflow as tf
import torch
# import torch.nn.functional as F
from unet_model import UNet, VENet, Unet
import numpy as np
import os
os.environ['PATH'] = 'E:\\backup\\software\\openslide-win64-20171122\\openslide-win64-20171122\\bin' + ';' + os.environ[
    'PATH']
import cv2
import copy
import scipy.ndimage as ndi
from skimage import morphology

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def samplemake(input1, locwhole, w, h, logs_dir, pred, batchsize, minsize):
    net = Unet(n_classes=1).cuda(0)
    net.load_state_dict(torch.load(logs_dir))
    net.eval()
    print("Model restored...")
    # ckpt = tf.train.get_checkpoint_state(logs_dir)
    # snapshot_path = '../model/{}'.format(args.model)
    # graph = tf.compat.v1.get_default_graph()
    # # print(graph)
    # image = graph.get_tensor_by_name('input_image:0')
    # annotationt = graph.get_tensor_by_name('annotation:0')
    # keep_probability = graph.get_tensor_by_name('keep_probability:0')
    # pred_annotation = graph.get_tensor_by_name('inference/prediction:0')
    # pred_annotation = tf.expand_dims(pred_annotation, dim=3)
    size1 = 400
    size2 = 400
    range1 = np.floor(w / size1)
    range2 = np.floor(h / size2)
    input = torch.zeros(batchsize, 3, size1, size2)
    # input = np.zeros([batchsize, size1, size2, 3])

    count = 0
    # model_annotations = np.zeros([batchsize, size1, size2])
    # model_annotations = np.expand_dims(model_annotations, axis=4)
    loc = np.zeros([batchsize, 2])
    loc1 = []
    order = 0
    for input11 in input1:
       # print(batchsize)
        input21 = np.ascontiguousarray(input11).transpose(2, 0, 1)
        patch = torch.from_numpy(input21).float()
        # print(patch.shape)
        input[count, :, :, :] = copy.deepcopy(patch)
        input = input.cuda(0)
        count = count + 1
        if count == batchsize:
            with torch.no_grad():
                mask_pred = net(input)
            predseg = (torch.sigmoid(mask_pred) > 0.5).float()
            predseg = predseg.unsqueeze(dim=1).cpu().numpy()
            # predseg = np.squeeze(predseg, axis=3)
            loc = locwhole[order*batchsize:(order+1)*batchsize]
            order += 1
            for num in range(batchsize):
                pred[np.int(loc[num][0]): np.int(loc[num][0] + size2),
                np.int(loc[num][1]): np.int(loc[num][1] + size1)] = copy.deepcopy(predseg[num, :, :])
                predseg1 = predseg[num, :, :]
                sum1 = len(predseg1[predseg1 == 0])
                if sum1 != size1 * size2:
                    loc1.append(loc[num])
            input = torch.zeros(batchsize, 3, size1, size2)
            count = 0
    if count != 0:
        with torch.no_grad():
            mask_pred = net(input)
        predseg = (torch.sigmoid(mask_pred) > 0.5).float()
        predseg = predseg.unsqueeze(dim=1).cpu().numpy()
        for num in range(count):
            loc = locwhole[order*batchsize:]
            pred[np.int(loc[num][0]): np.int(loc[num][0] + size2),
            np.int(loc[num][1]): np.int(loc[num][1] + size1)] = copy.deepcopy(predseg[num, :, :])
            predseg1 = predseg[num, :, :]
            sum1 = len(predseg1[predseg1 == 0])
            if sum1 != size1 * size2:
                loc1.append(loc[num])
    pp = copy.deepcopy(pred)
    bool1 = pp != 0
    bool2 = morphology.remove_small_objects(bool1, min_size=minsize, connectivity=2)
    pred = pred * 0
    pred[bool2] = 1
    pred = ndi.binary_fill_holes(pred)
    pred = np.array(pred, 'uint8')
    loc3 = []
    for i in range(len(loc1)):
        location = copy.deepcopy(loc1[i])
        #location[0] = location[0] - np.int(1/2 * size1)
        #location[1] = location[1] - np.int(1/2 * size2)
        loc3.append(location)

    loc2 = []
    locr2 = []
    for i in range(len(loc3)):
        location = loc3[i]
        #1
        location1 = copy.deepcopy(location)
        location1[0] = location1[0] - np.round(size1/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        location1 = copy.deepcopy(location)
        location1[1] = location1[1] - np.round(size2/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        location1 = copy.deepcopy(location)
        location1[0] = location1[0] - np.round(size1/2)
        location1[1] = location1[1] - np.round(size2/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        #2
        location1 = copy.deepcopy(location)
        location1[0] = location1[0] + np.round(size1/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        location1 = copy.deepcopy(location)
        location1[0] = location1[0] + np.round(size1/2)
        location1[1] = location1[1] - np.round(size2/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        #3
        location1 = copy.deepcopy(location)
        location1[1] = location1[1] + np.round(size2/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        location1 = copy.deepcopy(location)
        location1[0] = location1[0] - np.round(size1/2)
        location1[1] = location1[1] + np.round(size2/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        #4
        #3
        location1 = copy.deepcopy(location)
        location1[0] = location1[0] + np.round(size1/2)
        location1[1] = location1[1] + np.round(size2/2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
           loc2.append(location1)
        # 4
        # 3
        location1 = copy.deepcopy(location)
        location1[0] = location1[0] + np.round(size1 / 2)
        location1[1] = location1[1] + np.round(size2 / 2)
        if np.int(location1[0]) >= 0 and np.int(location1[1]) >= 0:
            loc2.append(location1)
        # location1 = copy.deepcopy(location)
        # location1[0] = np.max(location1[0] - np.round(size1/2)-np.round(size1/4), 0)
        # location1[1] = np.max(location1[1] - np.round(size2/2)-np.round(size1/4), 0)
        # if np.int(location1[0]) > 0 and np.int(location1[1]) > 0:
        #     loc2.append([location1, -1])
    return pred, loc2

def blockremove(lenth, batchsize, size1, size2, logs_dir, input3):
    net = Unet(n_classes=1).cuda(0)
    net.load_state_dict(torch.load(logs_dir))
    net.eval()
    predseg1 = np.zeros([lenth, size1, size2])
    count = 0
    input = torch.zeros(batchsize, 3, size1, size2)
    # input = np.zeros([batchsize, size1, size2, 3])
    num = 0
    for input31 in input3:
        input11 = np.ascontiguousarray(input31).transpose(2, 0, 1)
        patch = torch.from_numpy(input11).float()
        input[count, :, :, :] = copy.deepcopy(patch)
        input = input.cuda(0)
        count += 1
        if count == batchsize:
            with torch.no_grad():
                mask_pred = net(input)
            predseg = (torch.sigmoid(mask_pred) > 0.5).float()
            predseg = predseg.unsqueeze(dim=1).cpu().numpy()
            for ele in predseg:
               predseg1[num, :, :] = copy.deepcopy(ele)
               num += 1
            count = 0
            input = torch.zeros(batchsize, 3, size1, size2)
    if count != 0:
        with torch.no_grad():
            mask_pred = net(input)
        predseg = (torch.sigmoid(mask_pred) > 0.5).float()
        predseg = predseg.unsqueeze(dim=1).cpu().numpy()
        for ele in range(count):
            predseg1[num, :, :] = copy.deepcopy(predseg[ele, :, :])
    return predseg1

def writexml(xml_file, pred):
    xml_template = 'template.xml'
    tree = ET.parse(xml_template)

    annotations = tree.getroot()
    annotation = copy.deepcopy(annotations.find('Annotation'))
    regions = copy.deepcopy(annotation.find('Regions'))
    region = copy.deepcopy(regions.find('Region'))
    vertices = copy.deepcopy(region.find('Vertices'))
    vertex = copy.deepcopy(vertices.find('Vertex'))

    annotations.remove(annotations.find('Annotation'))
    annotation.remove(annotation.find('Regions'))
    regions.remove(regions.find('Region'))
    region.remove(region.find('Vertices'))
    vertices.remove(vertices.find('Vertex'))

    # 'colors' supports 8 layers at most
    colors = [[255, 255, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [128, 0, 255], [0, 128, 0], [255, 0, 255],
              [128, 128, 255]]
    img = pred
    color_map = {'1': 0, '2': 1, '3': 2}
    annotation_cnt = -1
    for id, val in color_map.items():

        tmp = np.zeros(img.shape, dtype=np.uint8)
        tmp[np.equal(img, val)] = val
        contours = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        annotation_cnt += 1
        color = colors[annotation_cnt]
        color = (color[2] * 65536 + color[1] * 256 + color[0])
        annotations.append(copy.deepcopy(annotation))
        cur_annotation = annotations.findall('Annotation')[-1]
        cur_annotation.set('Id', str(id))
        cur_annotation.set('LineColor', str(color))
        cur_annotation.append(copy.deepcopy(regions))
        cur_regions = cur_annotation.find('Regions')

        region_cnt = 0
        try:
            for contour in contours:
                region_cnt += 1
                cur_regions.append(copy.deepcopy(region))
                cur_region = cur_regions.findall('Region')[-1]
                cur_region.set('Id', str(region_cnt))
                cur_region.set('DisplayId', str(region_cnt))
                cur_region.append(copy.deepcopy(vertices))
                cur_vertices = cur_region.find('Vertices')

                for xy in contour:
                    vertex.set('X', str(xy[0, 0]))
                    vertex.set('Y', str(xy[0, 1]))
                    cur_vertices.append(copy.deepcopy(vertex))
        except TypeError:
            cd = 0

    tree.write(xml_file)
    """
    #sa1 = data.coins()
    import scipy.misc as misc
    import matplotlib.pylab as plt
    from skimage import color, filters, morphology
    sa1 = misc.imread('F:/sample.png')
    sa2 = color.rgb2gray(sa1)
    s1 = sa2.shape
    sa3 = np.zeros([s1[0], s1[1]])
    sa3[sa2 == 0] = 1
    #thresh = filters.threshold_otsu(sa3)
    #bw = morphology.closing(sa3>thresh, morphology.square(3))
    bw = sa3!=0
    dst = morphology.remove_small_objects(bw, min_size=1000, connectivity=1)
    sa4 = np.zeros_like(sa3, dtype=np.float64)
    sa4[dst] = 1
    plt.imshow(dst)
    plt.show()
    """
