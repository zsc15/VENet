import numpy as np
import glob
from svs_xml import glandseg
from svs_xml import colorcluster1
import scipy.misc as misc


def humanglandcluster(judgeLOC, judgeLOC1, judgeLOC2, Xml_file_name, svs_file, svmloc):
    jLOC = glob.glob(judgeLOC)
    jLOC1 = glob.glob(judgeLOC1)
    width = []
    width1 = []
    lenrecord = []
    for filename in jLOC:
        Data = np.load(filename)
        Data = Data.tolist()
        lenrecord.append(len(Data))
        for data in Data:
            width.append(data)
    for filename in jLOC1:
        Data = np.load(filename)
        Data = Data.tolist()
        for data in Data:
            width1.append(data)
    ordernum = np.load(judgeLOC2)
    ordernum = ordernum.tolist()
    dection = colorcluster1(Xml_file_name, svs_file, width, lenrecord, width1, svmloc, ordernum)
    return dection
if __name__ == "__main__":
    widthinfo = {}
    #judgeLOC = 'E:\\test2\\first\\test\\11\\*2.npy'
    judgeLOC = 'D:\\finaltest2\\*2.npy'
    jLOC = glob.glob(judgeLOC)
    #mainLOC = 'E:\\test2\\first\\test\\11\\*.svs'
    mainLOC = 'D:\\finaltest2\\*.svs'
    LOC = glob.glob(mainLOC)
    #xmainLOC = 'E:\\test2\\first\\test\\11\\*.xml'
    xmainLOC = 'D:\\finaltest2\\*.xml'
    #judgeLOC1 = 'E:\\test2\\first\\test\\11\\*1.npy'
    judgeLOC1 = 'D:\\finaltest2\\*1.npy'
    Xml_file_name = glob.glob(xmainLOC)
    for loc in LOC:
        loc1 = loc.split('.')[0] + '.npy'
        if loc1 not in jLOC:
            name = loc.split('.svs')
            xml_name = name[0] + '.xml'
            xmlA_name = name[0] + 'A.xml'
            width, appendixinfo = glandseg(loc, xml_name, xmlA_name)
            name1 = loc.split('\\')
            name1 = name1[-1].split('.svs')
            widthinfo[str(name1[0])] = width
            np.save(loc.split('.')[0]+'2'+'.npy', width)
            np.save(loc.split('.')[0]+'1'+'.npy', appendixinfo)
    dection = humanglandcluster(judgeLOC, judgeLOC1, Xml_file_name)
    data = open("E:\\test2\\first\\test\\11\\detection.txt", "w")
    for dec in dection:
        data.write(dec+'\n')
    data.close()