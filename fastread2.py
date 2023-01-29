import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
import scipy.ndimage as ndi
import multiprocessing
import shutil

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import createsample4_torch as cs
import numpy as np
import cv2
import os
import copy
import glob
import openslide
import time
import scipy.misc as misc
import joblib
from skimage import morphology
import imageio
# import sys
# sys.path.append('E:\\low differential\\TransforLearning_TensorFlow-master')
import TensorflowUtils as utils
from svs_xml import glandseg
from svs_xml import colorcluster1
from humandetect import humanglandcluster
from widthdect import measure
imagesize = 400
enlarge_size = 1500
def segmentblock(kclus, block, down):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(kclus.astype('uint8'))
    maxnum = np.max(labels)
    if maxnum > 0:
        for i in range(1, maxnum + 1):
            if stats[i, 4] > np.int((400/down)) ** 2:
                height_min = stats[i, 1]
                height_max = stats[i, 1] + stats[i, 3]
                width_min = stats[i, 0]
                width_max = stats[i, 0] + stats[i, 2]
                if (width_max - width_min + 1) > np.int((400/down)) and (height_max - height_min + 1) > np.int((400/down)):
                    if ((width_max - width_min) + 1) * ((height_max - height_min) + 1) < np.sum(kclus1[height_min:height_max, width_min:width_max]) * 2:
                        info = [height_min, height_max, width_min, width_max]
                        block.append(info)
                    else:
                        pp = np.zeros_like(labels, dtype='int')
                        pp[height_min: np.int(height_min + (height_max - height_min) / 2),
                        width_min: np.int(width_min + (width_max - width_min) / 2)] = 1
                        pp1 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp1*kclus, block, down))


                        pp = np.zeros_like(labels, dtype='int')
                        pp[np.int(height_min + (height_max - height_min) / 2): height_max,
                                  width_min: np.int(width_min + (width_max - width_min) / 2)] = 1
                        pp2 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp2*kclus, block, down))


                        pp = np.zeros_like(labels, dtype='int')
                        pp[height_min: np.int(height_min + (height_max - height_min) / 2),
                        np.int(width_min + (width_max - width_min) / 2): width_max] = 1
                        pp3 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp3*kclus, block, down))


                        pp = np.zeros_like(labels, dtype='int')
                        pp[np.int(height_min + (height_max - height_min) / 2):height_max,
                                  np.int(width_min + (width_max - width_min) / 2):width_max] = 1
                        pp4 = copy.deepcopy(pp)
                        block = copy.deepcopy(segmentblock(pp4*kclus, block, down))
    return block
def readimage(picloc, reimagesize, down, pad, svs_file, num, base, pkl_file):
    kclus1 = joblib.load(pkl_file)
    svs = openslide.OpenSlide(svs_file)
    if np.sum(kclus1[picloc[0]:picloc[0]+reimagesize, picloc[1]:picloc[1]+reimagesize]) >= np.int(reimagesize**2*3/4):
        region = np.array(svs.read_region((np.int(picloc[1]*down), np.int(picloc[0]*down)), 0, (400, 400)))
        region = copy.deepcopy(region[:, :, 0:3])
        code = cv2.imencode('.jpg', region)[1].tostring()
        if (picloc[0]-picloc[2]) % (4*pad) == 0 and (picloc[1]-picloc[3]) % (4*pad) == 0:
            # imageio.imwrite(base+str(num)+'.png', region)
            preimage = pre(region)
            preimage = normalize_img(preimage)######0-1归一化
            return [region, [np.int(picloc[0]*down), np.int(picloc[1]*down)], code, preimage, [picloc[0], picloc[1]]]
        else:
            return [region, [np.int(picloc[0]*down), np.int(picloc[1]*down)], code, 0, 0]
    else:
        return 0

def normalize_img(img):
    if np.max(img)!= np.min(img):
        trans_img = (img-np.min(img))/(np.max(img)-np.min(img))
        return trans_img
    elif np.max(img)>=1:
        # trans_img
        return img/np.max(img)
    else:
        return img

def saveimage(pic, num):
    misc.imsave('E:/testfile/' + str(num) + '.png', pic)

def pre(input1):
        input12 = utils.preprocess(input1)
        return input12

def postprocess(predseg1, pred1, loc2):
    if loc2[1] == -1:
       predseg1 = cv2.resize(predseg1, (1000, 1000))
    pred1[(pred1 == 0) & (predseg1 != 0)] = copy.deepcopy(predseg1[(pred1 == 0) & (predseg1 != 0)])
    pp = copy.deepcopy(pred1)
    bool1 = pp == 1
    bool2 = morphology.remove_small_objects(bool1, min_size=minsize, connectivity=4)
    pp1 = np.zeros_like(pp, dtype=np.float64)
    pp1[bool2] = 1
    pp1 = ndi.binary_fill_holes(pp1).astype(int)
    return [pp1, loc2]

def pre1(loc1, loc2, region):
    region = copy.deepcopy(region[:, :, 0:3])
    region_size = np.shape(region)[0]
    if region_size != 400:
       region = cv2.resize(region, (400, 400))
    input12 = utils.preprocess(region)
    if region_size == 400:
       return [input12, loc1, loc2]
    else:
       return[input12, loc1, loc2, -1]

def createpkl(svs_file, base):
    svs = openslide.OpenSlide(svs_file)
    nameorder = svs_file.split('/')[-1].split('.')[0]
    countlevel = svs.level_count
    dim1 = svs.level_dimensions[countlevel - 1]
    down = svs.level_downsamples[countlevel - 1]
    patch = np.array(svs.read_region((0, 0), countlevel - 1, (dim1[0], dim1[1])))
    patch = patch[:, :, 0:3]
    ac = copy.deepcopy(patch)
    ac1 = np.reshape(ac, [dim1[0] * dim1[1], 3], 'F')
    clus = copy.deepcopy(ac1)
    mod = KMeans(n_clusters=2, n_init=1)
    kclus = mod.fit_predict(clus)
    kclus1 = np.reshape(kclus, [dim1[1], dim1[0]], 'F')
    pan1 = np.sum((ac[:, :, 0] + ac[:, :, 1] + ac[:, :, 2]) * kclus1) / (np.sum(kclus1))
    pan2 = np.sum((ac[:, :, 0] + ac[:, :, 1] + ac[:, :, 2]) * (1 - kclus1)) / (dim1[0] * dim1[1] - np.sum(1 - kclus1))
    if pan2 < pan1:
        kclus1 = copy.deepcopy(1 - kclus1)
    kclus1p = copy.deepcopy(kclus1)
    bool1 = kclus1p != 0
    bool2 = morphology.remove_small_objects(bool1, min_size=np.int(50 ** 2 / (down ** 2)), connectivity=4)
    kclus1 = kclus1 * 0
    kclus1[bool2] = 1
    kclus1 = ndi.binary_fill_holes(kclus1)
    kclus1 = np.array(kclus1, 'uint8')
    kclus1 = copy.deepcopy(kclus1)
    kclus1 = np.array(kclus1, 'uint8')
    # imageio.imwrite('E:/1.png', kclus1 * 255)
    joblib.dump(kclus1, base + nameorder + '.pkl')
    return [kclus1, nameorder]

def widthcal(vertex_list, color_map, id, svs_file, img_dim,i):
    svs1 = openslide.OpenSlide(svs_file)
    xy = np.array([vertex_list])
    xy1 = np.squeeze(xy, 0)
    height_min = np.min(xy1[:, 1])
    height_max = np.max(xy1[:, 1])  # xml文件中第一列的位置数据为图片列的位置信息
    width_min = np.min(xy1[:, 0])
    width_max = np.max(xy1[:, 0])

    X = np.mean(xy1[:, 1])
    Y = np.mean(xy1[:, 0])
    height_min = np.max([height_min-50, 0])
    height_max = np.min([height_max+50, img_dim[1]])
    width_min = np.max([width_min - 50, 0])
    width_max = np.min([width_max+50, img_dim[0]])
    image_sample = svs1.read_region((width_min, height_min), 0, (
        width_max - width_min, height_max - height_min))
    Annotation_sample = np.zeros_like(image_sample)
    import matplotlib.pylab as plt
 # svs大图数据读取的第一条信息也是列的位置信息
    image_sample = np.array(image_sample)
    image_sample = image_sample[:, :, 0:3]
    Annotation_sample = Annotation_sample[:, :, 0]
    # for xx in xy1:
    #     annotation_sample[xx[1], xx[0]] = 255
    # plt.imshow(annotation_sample)
    # plt.show()
    xy1[:, 0] = xy1[:, 0] - width_min
    xy1[:, 1] = xy1[:, 1] - height_min
    xy1 = np.expand_dims(xy1, 0)
    annotation_sample = copy.deepcopy(Annotation_sample)
    cv2.fillPoly(annotation_sample, [xy1], color_map[id])
    xy1 = np.squeeze(xy1)
        # pp1 = copy.deepcopy(annotation_sample)
        # pp1[pp1 != 0] = 255
        # order += 1
        # misc.imsave('D:/test1/'+str(order)+'.png', pp1)
        # misc.imsave('D:\\2.png', image_sample)
    width = []
    appendixinfo = []
    calculateare = np.zeros_like(annotation_sample)
    calculateare[annotation_sample != 0] = 1
    areasquare = np.sum(calculateare)
    [avewidth, ratio, r, g, b, cof, nux, nuy] = measure(image_sample, annotation_sample)
    width.append([round(avewidth, 3), X, Y, round(ratio, 3), r, g, b, cof])
    if len(nux) != 0:
        assistloc = np.ones([len(nux), 1])
        for orderi in range(len(xy1)):
            X1 = (nux - xy1[orderi, 0]) ** 2 + (nuy - xy1[orderi, 1]) ** 2
            minloc = np.where(X1 == np.min(X1))[0][0]
            assistloc[minloc] = 0
        totalsum = np.sum(assistloc)
        appendixinfo.append([areasquare, totalsum])
       # region.set('Text', str(round(avewidth)))
    else:
        appendixinfo.append([areasquare, 0])
       # region.set('Text', str(round(avewidth)))
    #print(areasquare)
    return [width, appendixinfo, i]

if __name__ == '__main__':
    cpu_core = 18
    #
    # filebase = './finaltest2/'
    # filebase = '../backup/zsc/Nanfang_hosipital_data/test/'
    # filebase = '../Final_test/'
    filebase = '../test_svs/'
    files = sorted(glob.glob(filebase + '*.svs'), key=os.path.getmtime)
#    files = glob.glob(filebase+'*.svs')
    svmloc = './model/train_model2.m'
    filesname = []
    for file in files:
        name = file.split('./')[-1].split('/')[-1].split('.')[0]
        filesname.append(name)
        if os.path.isdir(filebase + name +'cache') == False:
           os.makedirs(filebase + name +'cache')
    Kclus1 = []
    Whole = []
    Nameorder = []
    pool = multiprocessing.Pool(processes=cpu_core)
    for svsname in filesname:
        if len(glob.glob(filebase + svsname +'cache/'+'*.pkl')) == 0:
            svs_file = filebase + svsname +'.svs'
            base = filebase + svsname +'cache/'
            Whole.append(pool.apply_async(createpkl, (svs_file, base, )))
    pool.close()
    pool.join()
    # whole = []
    # for i in range(len(Whole)):
    #     whole.append(Whole[i].get())
    # for i in range(len(whole)):
    #     Kclus1.append(whole[i][0])
    #     Nameorder.append(whole[i][1])
    # for i in range(len(Kclus1)):
    #     kclus1 = Kclus1[i]
    #     nameorder = Nameorder[i]
    #     joblib.dump(kclus1, base + nameorder + '.pkl')

    for svsnum in range(len(files)):
        # logs_dir = '../zsc/python_project/VENet_project/model/1_1_1_0VENet_nanfang_Lv_npy/'
        logs_dir = '../python_project/VENet_project/model/1_1_1_0UNet_nanfang_Lv_npy/best.pth'
        svs_file = files[svsnum] # svs_file,base
        nameorder = svs_file.split(filebase)[1].split('.')[0]
        base = filebase + nameorder + 'cache/'
        pkl_file = base + nameorder + '.pkl'
        kclus1 = joblib.load(pkl_file)
        xml_file1 = base + nameorder + '.xml'
        xml_file2 = base + nameorder + 'A.xml'
        if len(glob.glob(xml_file1)) == 0:
            #xml_file2 = 'E:/pictest/' + nameorder + '(2).xml'
            svs = openslide.OpenSlide(svs_file)
            dim = svs.dimensions
            countlevel = svs.level_count
            dim1 = svs.level_dimensions[countlevel - 1]
            down = svs.level_downsamples[countlevel - 1]
            pad = np.int(100 / down)
            shift = np.int(200 / down)
            reimagesize = np.int(400 / down)
            block = []
            blocks = segmentblock(kclus1, block, down)
            del kclus1
            results = []
            picloc = []
            pic = []
            picLoc = []
            for block in blocks:
                height_min = np.max([block[0] - shift, 0])
                height_max = np.min([block[1] + shift, dim1[1]])
                width_min = np.max([block[2] - shift, 0])
                width_max = np.min([block[3] + shift, dim1[0]])
                s1 = np.int((height_max-height_min-reimagesize)/pad)
                s2 = np.int((width_max-width_min-reimagesize)/pad)
                for i in range(s1):
                    for j in range(s2):
                        picloc.append([i*pad+height_min, j*pad+width_min, height_min, width_min])
            pool = multiprocessing.Pool(processes=cpu_core)
            for num in range(len(picloc)):
                pic.append(pool.apply_async(readimage, (picloc[num], reimagesize, down, pad, svs_file, num, base, pkl_file, )))
            pool.close()
            pool.join()
            del picloc
            figure = []
            whole = []
            codeimage = []
            preimage = []
            preloc = []
            for num in range(len(pic)):
                whole.append(pic[num].get())
            for num in range(len(whole)):
                if isinstance(whole[num], int) == False:
                    figure.append(whole[num][0])
                    picLoc.append(whole[num][1])
                    codeimage.append(whole[num][2])
                    if isinstance(whole[num][3], int) == False:
                        preimage.append(whole[num][3])
                        preloc.append(whole[num][1])
            del whole
            del figure
            del pic
            del codeimage
            del picLoc
            #detectlowregion(dim[0], dim[1], picLoc, xml_file1, xml_file2, codeimage)
            pred = np.zeros([dim[1], dim[0]], 'float32')
            batchsize = 20
            minsize = 50 * 50
            num = 0
            print(len(preimage))
            print('This way')
            print("Setting up Saver...")
            # device = torch.device('cuda:3')
            # net = Unet(n_classes=1).cuda(0)
            # net.load_state_dict(torch.load(logs_dir))
            # net.eval()
            # print("Model restored...")
            pred, loc2 = cs.samplemake(preimage, preloc, dim[0], dim[1], logs_dir, pred, batchsize, minsize)
            del preimage
            del preloc
            t5 = time.time()
            pool = multiprocessing.Pool(processes=cpu_core)  # 创建12个进程
            input1 = []
            for i in range(len(loc2)):
                if loc2[i][1] != -1:
                   region = np.array(svs.read_region(((np.int(loc2[i][1]), np.int(loc2[i][0]))), 0, (400, 400)))
                   input1.append(pool.apply_async(pre1, (np.int(loc2[i][1]), np.int(loc2[i][0]), region, )))
                else:
                   region = np.array(svs.read_region((np.int(loc2[i][0][1]), np.int(loc2[i][0][0])), 0, (1000, 1000)))
                   input1.append(pool.apply_async(pre1, (np.int(loc2[i][0][1]), np.int(loc2[i][0][0]), region, )))
            pool.close()
            pool.join()
            del region
            input2 = []
            loc2 = []
            for num in range(len(input1)):
                input2.append(input1[num].get())
            del input1
            input3 = []
            for ele in input2:
                input3.append(ele[0])
            for ele in input2:
                loc2.append(ele[1:])
            del input2
            print("Setting up Saver...block removing")
            # print("Model restored...")
            patch1 = cs.blockremove(len(loc2), batchsize, 400, 400, logs_dir, input3)
            del input3
            results2 = []
            pool2 = multiprocessing.Pool(processes= cpu_core)
            patchpred = []
            for i in range(len(loc2)):
                loc3 = loc2[i]
                if loc3[1] != -1:
                   patchpred.append(pred[np.int(loc3[1]):np.int(loc3[1]) + 400,
                         np.int(loc3[0]):np.int(loc3[0]) + 400])
                else:
                   patchpred.append(pred[np.int(loc3[1]):np.int(loc3[1]) + 1000,
                                     np.int(loc3[0]):np.int(loc3[0]) + 1000])
            for i in range(len(loc2)):
                results2.append(pool2.apply_async(postprocess, (patch1[i, :, :], patchpred[i], loc2[i], )))
            pool2.close()
            pool2.join()
            del patch1
            del loc2
            del patchpred
            for i in range(len(results2)):
                ele = results2[i].get()
                loc3 = ele[1]
                if loc3[1] != -1:
                   pred[np.int(loc3[1]):np.int(loc3[1]) + 400, np.int(loc3[0]):np.int(loc3[0]) + 400] = ele[0]
                else:
                   pred[np.int(loc3[0][1]):np.int(loc3[0][1]) + 1000, np.int(loc3[0][0]):np.int(loc3[0][0]) + 1000] = ele[0]
            #t5 = time.time()
            del results2
            pp = copy.deepcopy(pred)
            bool1 = pp != 0
            del pp
            bool2 = morphology.remove_small_objects(bool1, min_size=minsize, connectivity=4)
            pred = pred * 0
            pred[bool2] = 1
            del bool1
            del bool2
            pred[pred != 0] = 1
            pred = ndi.binary_fill_holes(pred)
            pred = np.array(pred, 'uint8')
            ######test
            # import imageio
            # imageio.imwrite(base+'mask.png', 255*pred)
            ##########
            cs.writexml(xml_file1, pred)
            t5 = time.time()
            del pred
            final = time.time()

        ###计算腺体各项特征指标
        widthinfo = {}
        judgeLOC = base + '2.npy'
        jLOC = glob.glob(judgeLOC)
        judgeLOC1 = base + '1.npy'
        judgeLOC2 = base + '3.npy'
        Inform = []
        print('start')
        #pool = multiprocessing.Pool(processes=10)  # 创建12个进程
        if len(glob.glob(judgeLOC1)) == 0 or len(glob.glob(judgeLOC)) == 0 or len(glob.glob(judgeLOC2)) == 0:
           color_map = {'2': 128, '3': 255}
           svs = openslide.OpenSlide(svs_file)
           img_dim = svs.dimensions
           width = []
           appendixinfo = []
           tree = ET.parse(xml_file1)
           annotations = tree.getroot()
           Vertex_list = []
           ordernum = []
           count = 1
           for annotation in annotations:
               id = annotation.get('Id')
               if id != '1':
                   print(id)
                   Id = id
                   for region in annotation.iter(tag='Region'):
                       vertex_list = []
                       for vertex in region.iter(tag='Vertex'):
                           vertex_list.append([int(np.floor(float(vertex.get('X')))), int(np.floor(float(vertex.get('Y'))))])
                       Vertex_list.append(vertex_list)
           #print('cao')
           pool = multiprocessing.Pool(processes=cpu_core)  # 创建12个进程
           for  i in range(len(Vertex_list)):
                vertex_list = Vertex_list[i]
                Inform.append(pool.apply_async(widthcal, (vertex_list, color_map, Id, svs_file, img_dim, i, )))
                # Inform.append(widthcal(vertex_list, color_map, Id, svs_file, img_dim,i))
          # tree.write(xml_file2)
           pool.close()
           pool.join()
           width = []
           appendixinfo = []
           for i in range(len(Inform)):
               ele = Inform[i].get()
               width.append(ele[0])
               appendixinfo.append(ele[1])
               ordernum.append(ele[2])
           name1 = svs_file.split('./')
           name1 = name1[-1].split('.svs')
           np.save(judgeLOC, np.array(width))
           np.save(judgeLOC1, np.array(appendixinfo))
           np.save(judgeLOC2, np.array(ordernum))
           width = np.load(judgeLOC)
           width = width.tolist()
           ordernum = np.load(judgeLOC2)
           ordernum = ordernum.tolist()
           tree = ET.parse(xml_file1)
           annotations = tree.getroot()
           num = 0
           for annotation in annotations:
               id = annotation.get('Id')
               if id != '1':
                   print(id)
                   Id = id
                   for region in annotation.iter(tag='Region'):
                       region.set('Text', str(round(width[ordernum[num]][0][0])))
                       num += 1
           tree.write(xml_file2)
        dection = humanglandcluster(judgeLOC, judgeLOC1, judgeLOC2, xml_file1, svs_file, svmloc)
        # svs_file, base
        shutil.move(svs_file, base+'/'+nameorder+'.svs')
        #data = open("E:\\test2\\first\\test\\11\\detection.txt", "w")
        #for dec in dection:
        #     data.write(dec + '\n')
        #data.close()
