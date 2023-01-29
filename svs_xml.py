
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import openslide
import numpy as np
import cv2
from sklearn.metrics import silhouette_score
import copy
import matplotlib.pylab as plt
from widthdect import measure
import scipy.misc as misc
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.externals import joblib
#------------------------------------------------------------------------------
# xml2img: convert Aperio ImageScope xml file to grayscale image
# 
# xml_file_name: string, the name of xml file
# img_file_name: string, the name of mark image file. PNG format is proposed
# img_dim: list or tuple, image file's shape, [num_of_columns, num_of_rows]
# color_map: dictionary, key is the layer id, value is the layer's grayscale value
#------------------------------------------------------------------------------
def xml2img(xml_file_name, img_file_name, img_dim, color_map):

    canvas = np.zeros([img_dim[1], img_dim[0]], dtype=np.uint8)

    tree = ET.parse(xml_file_name)
    annotations = tree.getroot()
    for annotation in annotations:
        id = annotation.get('Id')
        if id != '1':
            for region in annotation.iter(tag='Region'):
                vertex_list = []
                for vertex in region.iter(tag='Vertex'):
                    vertex_list.append([int(vertex.get('X')), int(vertex.get('Y'))])

                xy = np.array([vertex_list])
                cv2.fillPoly(canvas, xy, color_map[id])

    #cv2.imwrite(img_file_name, canvas)

    return canvas

def xml2img1(xml_file_name, index, img_dim, color_map, ordernum1):

    canvas = np.zeros([img_dim[1], img_dim[0]], dtype=np.uint8)
    print(xml_file_name)
    tree = ET.parse(xml_file_name)
    annotations = tree.getroot()
    ordernum = 0
    for annotation in annotations:
        id = annotation.get('Id')
        if id != '1':
            for region in annotation.iter(tag='Region'):
                if index[ordernum1[ordernum]] != 0:
                    vertex_list = []
                    for vertex in region.iter(tag='Vertex'):
                        vertex_list.append([int(vertex.get('X')), int(vertex.get('Y'))])

                    xy = np.array([vertex_list])
                    cv2.fillPoly(canvas, xy, color_map[str(index[ordernum,0])])
                ordernum += 1
    return canvas

def measurewidth(xml_file_name, svs, img_dim, color_map, xmlA_name):
    order = 0
    canvas = np.zeros([img_dim[1], img_dim[0]], dtype = np.uint8)
    width = []
    appendixinfo = []
    tree = ET.parse(xml_file_name)
    annotations = tree.getroot()
    for annotation in annotations:
        id = annotation.get('Id')
        if id != '1':
            for region in annotation.iter(tag='Region'):
                vertex_list = []
                for vertex in region.iter(tag='Vertex'):
                    vertex_list.append([int(np.floor(float(vertex.get('X')))), int(np.floor(float(vertex.get('Y'))))])

                xy = np.array([vertex_list])
                canvas1 = copy.deepcopy(canvas)
                cv2.fillPoly(canvas1, xy, color_map[id])
                xy1 = np.squeeze(xy, 0)
                height_min = np.min(xy1[:, 1])
                height_max = np.max(xy1[:, 1])     # xml文件中第一列的位置数据为图片列的位置信息
                width_min = np.min(xy1[:, 0])
                width_max = np.max(xy1[:, 0])
                areasquare = (height_max-height_min)*(width_max - width_min)
                X = np.mean(xy1[:, 1])
                Y = np.mean(xy1[:, 0])
                if height_min - 50 >0 and width_min - 50 >0 and height_max + 50 < img_dim[1] and width_max + 50 < img_dim[0]:
                    annotation_sample = canvas1[(height_min - 50): (height_max + 50), (width_min - 50): (width_max + 50)]
                    image_sample = svs.read_region((width_min - 50, height_min - 50), 0, (width_max-width_min+100, height_max-height_min+100)) #svs大图数据读取的第一条信息也是列的位置信息
                    image_sample = np.array(image_sample)
                    image_sample = image_sample[:, :, 0:3]
                    # pp1 = copy.deepcopy(annotation_sample)
                    # pp1[pp1 != 0] = 255
                    # order += 1
                    # misc.imsave('D:/test1/'+str(order)+'.png', pp1)
                    #misc.imsave('D:\\2.png', image_sample)
                    [avewidth, ratio, r, g, b, cof] = measure(image_sample, annotation_sample)
                    width.append([round(avewidth, 3), X, Y, round(ratio, 3), r, g, b, cof, nux, nuy])
                    if len(nux) != 0:
                        assistloc = np.ones([len(nux), 1])
                        for orderi in range(len(xy1)):
                            X1 = (nux - xy1[orderi, 0])**2 + (nuy - xy1[orderi, 1])**2
                            minloc = np.where(X1 == np.min(X1))[0][0]
                            assistloc[minloc] = 0
                        totalsum = sum(assistloc)
                        appendixinfo.append([areasquare, totalsum])
                        region.set('Text', str(round(avewidth)))
                    else:
                        appendixinfo.append([areasquare, 0])
                        region.set('Text', str(round(avewidth)))
                    #print(len(nux))
                else:
                    annotation_sample = canvas1[(height_min): (height_max ), (width_min ): (width_max )]
                    image_sample = svs.read_region((width_min, height_min), 0, (width_max-width_min, height_max-height_min)) #svs大图数据读取的第一条信息也是列的位置信息
                    image_sample = np.array(image_sample)
                    image_sample = image_sample[:, :, 0:3]
                    # pp1 = copy.deepcopy(annotation_sample)
                    # pp1[pp1 != 0] = 255
                    # order += 1
                    # misc.imsave('D:/test1/'+str(order)+'.png', pp1)
                    #misc.imsave('D:\\2.png', image_sample)
                    [avewidth, ratio, r, g, b, cof, nux, nuy] = measure(image_sample, annotation_sample)
                    width.append([round(avewidth, 3), X, Y, round(ratio, 3), r, g, b, cof])
                    if len(nux) != 0:
                        assistloc = np.ones([len(nux), 1])
                        for orderi in range(len(xy1)):
                            X1 = (nux - xy1[orderi, 0])**2 + (nuy - xy1[orderi, 1])**2
                            minloc = np.where(X1 == np.min(X1))[0][0]
                            assistloc[minloc] = 0
                        totalsum = sum(assistloc)
                        appendixinfo.append([areasquare, totalsum])
                        region.set('Text', str(round(avewidth)))
                    else:
                        appendixinfo.append([areasquare, 0])
                        region.set('Text', str(round(avewidth)))
                    #print(len(nux))
    tree.write(xmlA_name)
    return width, appendixinfo
def colorcluster(xml_file_name, width, img_dim):
    width1 = []
    for i in range(len(width)):
        width[i][1] = round(width[i][1], 3)
        width[i][2] = round(width[i][2], 3)
        width1.append(width[i][0])

    tree = ET.parse(xml_file_name)
    annotations = tree.getroot()
    sigma = np.std(width1)
    ave = np.mean(width1)
    width11 = np.array(width1)
    width11[width11 > 30] = 0
    for i in range(len(width)):
        width[i][0] = copy.deepcopy(width11[i])
    maxnum = np.max(width11)
    loc = width1.index(maxnum)
    #
    # SSE = []
    # for k in range(1, 51):
    #     mod = KMeans(n_clusters=k)
    #     avewidth = [[i] for i in width11]
    #     kclus = mod.fit_predict(avewidth)
    #     SSE.append(mod.inertia_)
    # X = range(1, len(SSE)+1)
    # import matplotlib.pylab as plt
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # plt.plot(X, np.log(SSE), 'o-')
    # return SSE
    for k in range(1, len(width1)+1):
        mod = KMeans(n_clusters=k)
        avewidth = [[i] for i in width11]
        kclus = mod.fit_predict(avewidth)
        num = len(kclus == kclus[loc])
        if num > round(len(width11)*0.05) and num < round(len(width11)*0.45):
           clus = k
           break
    mod = KMeans(n_clusters=clus)
    avewidth = [[i] for i in width11]
    kclus = mod.fit_predict(avewidth)
    index = np.where(kclus == kclus[loc])
    index = index[0]
    avewidth1 = []
    for index1 in index:
        avewidth1.append([width[index1][1], width[index1][2]])
    # SSE = []
    # coef = []
    # for k in range(2, 51):
    #     mod = KMeans(n_clusters=k)
    #     kclus = mod.fit_predict(avewidth1)
    #     SSE.append(mod.inertia_)
    #     silhoutte_avg = silhouette_score(avewidth1, kclus)
    #     coef.append(silhoutte_avg)
    mod = KMeans(n_clusters=5)
    kclus1 = mod.fit_predict(avewidth1)
    INDICE = copy.deepcopy(kclus)
    INDICE[kclus != kclus[loc]] = 0
    for i in range(0, 5):
        INDICE[index[kclus1 == i]] = i + 1
    color_map = {}
    interval = round(255/(5+1))
    for i in range(1, 5+1):
        color_map[str(i)] = interval * i
    pred = xml2img1(xml_file_name, INDICE, img_dim, color_map)
    splitname = xml_file_name.split('.xml')
    xml_file_name1 = splitname[0] + '1' + '.xml'
    img2xml1(pred, xml_file_name1, color_map)

def colorcluster1(xml_file_name, svs_file, width, lenrecord, appendix,svmloc,ordernum):
    width = np.squeeze(width)
    appendix = np.squeeze(appendix)
    print(width.shape)
    width = width.tolist()
    appendix = appendix.tolist()
    svmclassifier = joblib.load(svmloc)
    width1 = []
    area = []
    nucle = []
    print(width)
    for i in range(len(width)):
        width[i][1] = round(width[i][1], 3)
        width[i][2] = round(width[i][2], 3)
        width1.append(width[i][0])
    for i in range(len(width)):
        area.append(appendix[i][0])
        nucle.append(appendix[i][1])
    sigma = np.std(width1)
    ave = np.mean(width1)
    width11 = np.array(width1)
    width11[width11 > 50] = 0
    for i in range(len(width)):
        width[i][0] = copy.deepcopy(width11[i])
    judge = []
    INDICE = np.zeros([len(width), 1], dtype='uint8')
    num = 0
    for features in width:
        feature = np.zeros([1, 3])
        feature[0, 0] = features[0]
        feature[0, 1] = features[3]
        feature[0, 2] = features[7]
        index = svmclassifier.predict(feature)
        if (index == 1 and feature[0,0]!=0):
           INDICE[num] = 1
        num += 1
    color_map = {'1': 128, '2': 255}
    INDICE1 = copy.deepcopy(INDICE)
    order = 0
    detection = []
    widthsample = copy.deepcopy(width)
    areasample = copy.deepcopy(area)
    nuclesample = copy.deepcopy(nucle)
    svs = openslide.OpenSlide(svs_file)
    img_dim = svs.dimensions
    INDICE2 = copy.deepcopy(INDICE1[0:lenrecord[order]])
    newwidth = copy.deepcopy(widthsample[0:lenrecord[order]])
    del widthsample[0: lenrecord[order]]
    newarea = copy.deepcopy(areasample[0:lenrecord[order]])
    del areasample[0: lenrecord[order]]
    del nuclesample[0: lenrecord[order]]
    order += 1
    #INDICE2 = INDICE2 + IINDICE2
    #INDICE2[INDICE2 > 0] = 1
    if  len(INDICE2) >= 3 and np.sum(INDICE1) !=0:
        avewidth1 = []
        for i in range(len(INDICE2)):
            if INDICE2[i] != 0:
                avewidth1.append([newwidth[i][1], newwidth[i][2]])
        Z = linkage(np.array(avewidth1), method='single', metric='Euclid')
        kclus = fcluster(Z, 600, 'distance')
        k = np.max(kclus)
        kclus = kclus - 1
        INDICE22 = copy.deepcopy(INDICE2)
        for i in range(k):
            if len(np.where(kclus == i)[0]) < 4:
                kclus[kclus == i] = -1
        loc1 = np.where(kclus == -1)[0]
        judgement = 0
        if len(loc1) != 0:
                loc = np.where(INDICE2 == 1)[0]
                hloc = loc[loc1]
                for ele in hloc:
                    if newarea[ele] > 90000:
                        judgement = 1
                if judgement == 0:
                   INDICE2[loc[loc1]] = 0
        if len(INDICE2) > 0:
            Pred = xml2img1(xml_file_name, INDICE2, img_dim, color_map,ordernum)
            INDICE3 = copy.deepcopy(INDICE22)
            HULL = []
            for i in range(k):
                if len(np.where(kclus == i)[0]) > 0:
                    loc1 = np.where(kclus != i)[0]
                    loc = np.where(INDICE22 == 1)[0]
                    INDICE3[loc[loc1]] = 0
                    pred = xml2img1(xml_file_name, INDICE3, img_dim, color_map, ordernum)
                    INDICE3 = copy.deepcopy(INDICE22)
                    contours = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    cnt = contours[0]
                    location = []
                    for cnt1 in cnt:
                        for cnt2 in cnt1:
                            location.append(cnt2)
                    location = np.array(location)
                    print(len(location))
                    hull = cv2.convexHull(location)
                    HULL.append(hull)
            splitname = xml_file_name.split('.xml')
            # print(splitname)
            order = splitname[0].split('/')[-1]
            # object_path = '../xml_file/'+order+'.xml'
            xml_file_name0 = '../xml_file/'+order+'.xml'
            xml_file_name1 = splitname[0] + '1' + '.xml'
            img2xml1(Pred, xml_file_name0, color_map, HULL)
            img2xml1(Pred, xml_file_name1, color_map, HULL)
            detection.append('There may be ill in ' + str(svs_file))
    else:
        splitname = xml_file_name.split('.xml')
        order = splitname[0].split('/')[-1]
        xml_file_name0 = '../xml_file/' + order + '.xml'
        xml_file_name1 = splitname[0] + '1' + '.xml'
        img2xmlNULL(xml_file_name0)
        img2xmlNULL(xml_file_name1)
    return detection
    #avewidth1 = []
    # for index1 in index:
    #     avewidth1.append([width[index1][1], width[index1][2]])
    # SSE = []
    # coef = []
    # for k in range(2, 51):
    #     mod = KMeans(n_clusters=k)
    #     kclus = mod.fit_predict(avewidth1)
    #     SSE.append(mod.inertia_)
    #     silhoutte_avg = silhouette_score(avewidth1, kclus)
    #     coef.append(silhoutte_avg)
    # mod = KMeans(n_clusters=5)
    # kclus1 = mod.fit_predict(avewidth1)
    # INDICE = copy.deepcopy(kclus)
    # INDICE[kclus != kclus[loc]] = 0
    # for i in range(0, 5):
    #     INDICE[index[kclus1 == i]] = i + 1
    # color_map = {}
    # interval = round(255 / (5 + 1))
    # for i in range(1, 5 + 1):
    #     color_map[str(i)] = interval * i
    # pred = xml2img1(xml_file_name, INDICE, img_dim, color_map)
    # splitname = xml_file_name.split('.xml')
    # xml_file_name1 = splitname[0] + '1' + '.xml'
    # img2xml1(pred, xml_file_name1, color_map)

#------------------------------------------------------------------------------
# img2xml: generate Aperio ImageScope xml file from grayscale mark image
#
# img_file_name: string, the name of mark image file
# xml_file_name: string, the name of xml file
# color_map: dictionary, key is the layer id, value is the layer's grayscale value
#------------------------------------------------------------------------------
def img2xml(img_file_name, xml_file_name, color_map):

    xml_template = 'x:/template.xml'
    tree = ET.parse(xml_template)

    annotations = tree.getroot()
    annotation = copy.deepcopy( annotations.find('Annotation') )
    regions = copy.deepcopy( annotation.find('Regions') )
    region = copy.deepcopy( regions.find('Region') )
    vertices = copy.deepcopy( region.find('Vertices') )
    vertex = copy.deepcopy( vertices.find('Vertex') )

    annotations.remove( annotations.find('Annotation') )
    annotation.remove( annotation.find('Regions') )
    regions.remove( regions.find('Region') )
    region.remove( region.find('Vertices') )
    vertices.remove( vertices.find('Vertex') )

    # 'colors' supports 8 layers at most
    colors = [[255,255,0], [0,255,0], [0,0,255], [255,0,0], [128,0,255], [0,128,0], [255,0,255], [128,128,255]]

    img = cv2.imread(img_file_name, cv2.IMREAD_UNCHANGED)
    
    annotation_cnt = -1
    for id, val in color_map.items():

        tmp = np.zeros(img.shape, dtype=np.uint8)
        tmp[np.where(np.equal(img, val))] = 255
        contours = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
        contours = contours[1]

        annotation_cnt += 1
        color = colors[annotation_cnt]
        color = (color[2]*65536 + color[1]*256 + color[0])
        annotations.append(copy.deepcopy(annotation))
        cur_annotation = annotations.findall('Annotation')[-1]
        cur_annotation.set('Id', str(id))
        cur_annotation.set('LineColor', str(color))
        cur_annotation.append(copy.deepcopy(regions))
        cur_regions = cur_annotation.find('Regions')

        region_cnt = 0
        for contour in contours:
            region_cnt += 1
            cur_regions.append( copy.deepcopy(region) )
            cur_region = cur_regions.findall('Region')[-1]
            cur_region.set('Id', str(region_cnt))
            cur_region.set('DisplayId', str(region_cnt))
            cur_region.append( copy.deepcopy(vertices) )
            cur_vertices = cur_region.find('Vertices')

            for xy in contour:
                vertex.set('X', str(xy[0,0]))
                vertex.set('Y', str(xy[0,1]))
                cur_vertices.append( copy.deepcopy(vertex) )
    tree.write(xml_file_name)
    return


def img2xml1(img, xml_file_name, color_map, HULL):
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
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [128, 0, 255], [0, 128, 0], [255, 0, 255],
              [128, 128, 255]]
    annotation_cnt = -1
    for id, val in color_map.items():

        tmp = np.zeros(img.shape, dtype=np.uint8)
        tmp[np.where(np.equal(img, val))] = 255
        contours = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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
        # for contour in contours:
        #     region_cnt += 1
        #     cur_regions.append(copy.deepcopy(region))
        #     cur_region = cur_regions.findall('Region')[-1]
        #     cur_region.set('Id', str(region_cnt))
        #     cur_region.set('DisplayId', str(region_cnt))
        #     cur_region.append(copy.deepcopy(vertices))
        #     cur_vertices = cur_region.find('Vertices')
        #
        #     for xy in contour:
        #         vertex.set('X', str(xy[0, 0]))
        #         vertex.set('Y', str(xy[0, 1]))
        #         cur_vertices.append(copy.deepcopy(vertex))
        for hull1 in HULL:
            region_cnt += 1
            cur_regions.append(copy.deepcopy(region))
            cur_region = cur_regions.findall('Region')[-1]
            cur_region.set('Id', str(region_cnt))
            cur_region.set('DisplayId', str(region_cnt))
            cur_region.append(copy.deepcopy(vertices))
            cur_vertices = cur_region.find('Vertices')
            for xy in hull1:
                xy = xy[0]
                vertex.set('X', str(xy[0]))
                vertex.set('Y', str(xy[1]))
                cur_vertices.append(copy.deepcopy(vertex))
    tree.write(xml_file_name)
    return


def img2xmlNULL(xml_file_name):
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
    tree.write(xml_file_name)
    return

# def convexhull(xml_file_name, Hull):
#     tree = ET.parse(xml_file_name)
#
#     annotations = tree.getroot()
#     annotation = copy.deepcopy(annotations.find('Annotation'))
#     regions = copy.deepcopy(annotation.find('Regions'))
#     region = copy.deepcopy(regions.find('Region'))
#     vertices = copy.deepcopy(region.find('Vertices'))
#     vertex = copy.deepcopy(vertices.find('Vertex'))
#     id = 20
#     # 'colors' supports 8 layers at most
#     colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [128, 0, 255], [0, 128, 0], [255, 0, 255],
#               [128, 128, 255]]
#     color = (colors[0][2] * 65536 + colors[0][1] * 256 + colors[0][0])
#     annotations.append(copy.deepcopy(annotation))
#     cur_annotation = annotations.findall('Annotation')[-1]
#     cur_annotation.set('Id', str(id))
#     cur_annotation.set('LineColor', str(color))
#     cur_annotation.append(copy.deepcopy(regions))
#     cur_regions = cur_annotation.find('Regions')
#     region_cnt = 0
#     for hull1 in HULL:
#         region_cnt += 1
#         cur_regions.append(copy.deepcopy(region))
#         cur_region = cur_regions.findall('Region')[-1]
#         cur_region.set('Id', str(region_cnt))
#         cur_region.set('DisplayId', str(region_cnt))
#         cur_region.append(copy.deepcopy(vertices))
#         cur_vertices = cur_region.find('Vertices')
#         for xy in hull1:
#             xy = xy[0]
#             vertex.set('X', str(xy[0]))
#             vertex.set('Y', str(xy[1]))
#             cur_vertices.append(copy.deepcopy(vertex))
#     tree.write(xml_file_name)
#     return

### test codes
def glandseg(svs_file_name, xml_file_name, xmlA_name):
    #svs_file_name = 'E:\\cancer test\\Wu Rui-Bo\\1616198.svs'
    #xml_file_name = 'E:\\cancer test\\Wu Rui-Bo\\1616198.xml'
    color_map = {'2':128, '3':255}
    svs = openslide.OpenSlide(svs_file_name)
    img_dim = svs.dimensions
    #mark = xml2img(xml_file_name, img_file_name, img_dim, color_map)
    WIDTH, Appendixinfo = measurewidth(xml_file_name, svs, img_dim, color_map, xmlA_name)
    #colorcluster(xml_file_name, WIDTH, img_dim)
    return WIDTH, Appendixinfo