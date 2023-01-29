import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import copy

def SeparateStains(imageRGB, matrix):
   imageRGB = np.float64(imageRGB)+2
   imagesize = imageRGB.shape
   B = np.reshape(-np.log(imageRGB/257), [imagesize[0]*imagesize[1], 3], 'F')
   for i in range(imagesize[0]*imagesize[1]):
       if np.sum(B[i, :])/3 < 0.25 or B[i, 0] < 0.2 or B[i, 1] < 0.2 or B[i, 2] < 0.2:
           B[i, :] = 0
   imageout = B * matrix
   imageout[imageout < 0] = 0
   imageout = np.reshape(np.array(imageout), [imagesize[0], imagesize[1], 3], 'F')
   return imageout

def widthdetect(area1, area2, line1):
    LOC = []
    imagesize = line1.shape
    X = imagesize[0]
    Y = imagesize[1]
    line11 = np.reshape(line1, [X*Y], 'F')
    loc1 = np.where((line11 == 1))
    loc1 = loc1[0]
    # LOCC = []
    # i = 0
    # while i <= len(loc1)-1:
    #     LOCC.append(loc1[i])
    #     i += 10
    # loc1 = np.squeeze(np.array(LOCC))
    loc1list = np.zeros([2, len(loc1)], 'int32')

    for i in range(len(loc1)):
        loc1list[0, i] = loc1[i] % X
        loc1list[1, i] = np.floor(loc1[i] / X) + 1
    area11 = np.reshape(area1, [X * Y], 'F')
    area21 = np.reshape(area2, [X * Y], 'F')
    loc2 = np.where(area11 == 1)
    loc2 = loc2[0]
    loc3 = np.where(area21 == 1)
    loc3 = loc3[0]
    width = []
    Record = np.zeros([len(loc1), 3])
    for i in range(len(loc1)):
        locrecord = np.zeros([1, 2])
        record = 0
        diffa = loc1list[0, :] - loc1list[0, i]
        diffb = loc1list[1, :] - loc1list[1, i]
        diffsquare = diffa**2 + diffb**2
        diff_max = diffsquare <= 50
        diff_min = diffsquare > 0
        diff_rst = diff_max & diff_min
        locnum = np.where(diff_rst == True)
        locnum = locnum[0]
        Record[i, 1] = np.sum((loc1list[0, locnum] - loc1list[0, i]) ** 2)
        Record[i, 2] = np.sum((loc1list[1,  i] - loc1list[1, locnum]) ** 2)
        if Record[i, 1] != 0 and Record[i, 2] != 0:
           data = np.zeros([len(locnum), 2])
           data[:, 0] = loc1list[0, locnum]
           data[:, 1] = loc1list[1, locnum]
           dataHomo = copy.deepcopy(data)
           dataHomo[:, 0] -= np.sum(data[:, 0]) / len(locnum)
           dataHomo[:, 1] -= np.sum(data[:, 1]) / len(locnum)
           # data matrix
           dataMatrix = np.dot(dataHomo.transpose(), dataHomo)
           u, s, vh = np.linalg.svd(dataMatrix, full_matrices=True)
           n = u[:, -1]
           if n[0]!=0:
              k = n[1]/n[0]
           else:
              k = n[1]/(10**(-6))

          # b = np.sum(data[:, 1]) / len(locnum) - k * np.sum(data[:, 0]) / len(locnum)
          # print(k)
          # k = np.sum((loc1list[0, locnum] - loc1list[0, i]) ** 2) / (np.sum((loc1list[1, i] - loc1list[1, locnum]) * (loc1list[0, locnum] - loc1list[0, i])))
           #k = interplolate.linear(loc1list[0, i], loc1list[1, i])
          # k = -1/k
           #print(k)
           #k, b = optimize.curve_fit(f_1, loc1list[0, locnum], loc1list[1, locnum])[0]
           #k = -1/k
           #print(k)
           Record[i, 0] = k
           judge = np.where(loc3 == (loc1list[0, i] + 1) + (round(k + loc1list[1, i]) - 1) * X)
           if len(judge[0]) != 0:
              for j in range(loc1list[0, i], X+1):
                  judge = np.where(loc2 == j+(round(k * (j - loc1list[0, i]) + loc1list[1, i]) - 1) * X)
                  if len(judge[0]) != 0:
                     LOC.append(j + (round(k * (j - loc1list[0, i]) + loc1list[1, i]) - 1)*X)
                     locrecord[0, record] = j
                     record = record + 1
                     for j1 in range(j+1, X+1):
                         judge = np.where(loc2 == j1 + (round(k * (j1 - loc1list[0, i]) + loc1list[1, i]) - 1) * X)
                         if len(judge[0]) == 0:
                            locrecord[0, record] = j1 - 1
                            record = record + 1
                            break
                         LOC.append(j1 + (round(k * (j1 - loc1list[0, i]) + loc1list[1, i]) - 1) * X)
                  if record == 2:
                     width.append(abs(locrecord[0, 0] - locrecord[0, 1]) * np.sqrt(1 + k**2))
                     break
                  if j == X and record == 1:
                     width.append(abs(locrecord[0, 0] - X) * np.sqrt(1 + k**2))
                  if j == X and record == 0:
                     width.append(0)
           else:
              for j in range(loc1list[0, i], 0, -1):
                  judge = np.where(loc2 == j + (round(k * (j - loc1list[0, i]) + loc1list[1, i]) - 1) * X)
                  if len(judge[0]) != 0:
                     locrecord[0, record] = j
                     record = record + 1
                     LOC.append(j+(round(k * (j - loc1list[0, i]) + loc1list[1, i]) - 1)*X)
                     for j1 in range(j-1, 0, -1):
                         judge = np.where(loc2 == j1 + (round(k * (j1 - loc1list[0, i]) + loc1list[1, i]) - 1) * X)
                         if len(judge[0]) == 0:
                            locrecord[0, record] = j1+1
                            record = record + 1
                            break
                         LOC.append(j1 + (round(k * (j1 - loc1list[0, i]) + loc1list[1, i]) - 1) * X)
                  if record == 2:
                     width.append(abs(locrecord[0, 0] - locrecord[0, 1]) * np.sqrt(1 + k**2))
                     break
                  if j == 1 and record == 1:
                     width.append(abs(locrecord[0, 0] - 1) * np.sqrt(1 + k**2))
                  if j == 1 and record == 0:
                     width.append(0)

        if Record[i, 2] == 0 and Record[i, 1] != 0:
           Record[i, 0] = 0
           judge = np.where(loc3 == loc1list[0, i] + loc1list[1, i] * X)
           if len(judge[0]) != 0:
              for j in range(loc1list[1, i], Y+1):
                  judge = np.where(loc2 == loc1list[0, i] + (j - 1) * X)
                  if len(judge[0]) != 0:
                     LOC.append(loc1list[0, i] + (j - 1)*X)
                     locrecord[0, record] = j
                     record = record + 1
                     for j1 in range(j+1, Y+1):
                         judge = np.where(loc2 == loc1list[0, i] + (j1 - 1) * X)
                         if len(judge[0]) == 0:
                            locrecord[0, record] = j1 - 1
                            record = record + 1
                            break
                         LOC.append(loc1list[0, i] + (j1 - 1) * X)
                  if record == 2:
                     width.append(abs(locrecord[0, 0] - locrecord[0, 1]))
                     break
                  if j == Y and record == 1:
                     width.append(locrecord[0, 0] - Y)
                  if j == Y and record == 0:
                     width.append(0)
           else:
              for j in range(loc1list[1, i],0,-1):
                  judge = np.where(loc2 == loc1list[0, i] + (j - 1) * X)
                  if len(judge[0]) != 0:
                     LOC.append(loc1list[0, i] + (j - 1) * X)
                     locrecord[0, record] = j
                     record = record + 1
                     for j1 in range(j-1,0,-1):
                         judge = np.where(loc2 == loc1list[0, i] + (j1 - 1) * X)
                         if len(judge[0]) == 0:
                             locrecord[0, record] = j1 + 1
                             record = record + 1
                             break
                         LOC.append(loc1list[0, i] + (j1 - 1) * X)
                  if record == 2:
                     width.append(abs(locrecord[0, 0] - locrecord[0, 1]))
                     break
                  if j == 1 and record == 1:
                     width.append(locrecord[0, 0] - 1)
                  if j == 1 and record == 0:
                     width.append(0)
        if Record[i, 1] == 0:
           Record[i, 0] = 0
           judge = np.where(loc3 == (loc1list[0, i] + 1) + (loc1list[1, i] - 1) * X)
           if len(judge[0]) != 0:
              for j in range(loc1list[0, i], X+1):
                  judge = np.where(loc2 == j + (loc1list[1, i] - 1) * X)
                  if len(judge[0]) != 0:
                     LOC.append(j + (loc1list[1, i] - 1) * X)
                     locrecord[0, record] = j
                     record = record + 1
                     for j1 in range(j+1, X+1):
                         judge = np.where(loc2 == j1 + (loc1list[1, i] - 1) * X)
                         if len(judge[0]) == 0:
                            locrecord[0, record] = j1-1
                            record = record + 1
                            break
                         LOC.append(j1 + (loc1list[1, i] - 1) * X)
                  if record == 2:
                     width.append(abs(locrecord[0, 0] - locrecord[0, 1]))
                     break
                  if j == X and record == 1:
                     width.append(abs(locrecord[0, 0] - X))
                  if j == X and record == 0:
                     width.append(0)
           else:
              for j in range(loc1list[0, i], 0, -1):
                  judge = np.where(loc2 == j + (loc1list[1, i] - 1) *X)
                  if len(judge[0]) != 0:
                     LOC.append(j + (loc1list[1, i] - 1) * X)
                     locrecord[0, record] = j
                     record = record + 1
                     for j1 in range(j-1, 0, -1):
                         judge = np.where(loc2 == j1 + (loc1list[1, i]-1)*X)
                         if len(judge[0]) == 0:
                            locrecord[0, record] = j1 + 1
                            record = record + 1
                            break
                         LOC.append(j1 + (loc1list[1, i]-1)*X)
                  if record == 2:
                     width.append(abs(locrecord[0, 0] - locrecord[0, 1]))
                     break
                  if j == 1 and record == 1:
                     width.append(locrecord[0, 0]-1)
                  if j == 1 and record == 0:
                     width.append(0)
    return width, LOC


def preprocess1(input):
    #input = copy.deepcopy(valid_images)
    input = copy.deepcopy(np.float64(input))
    s = input.shape
    if len(s) == 4:
        for i in range(s[0]):
            input1 = copy.deepcopy(input[i, :, :, :])
            for j in range(s[-1]):
                b = copy.deepcopy(input1[:, :, j])
                b1 = b - np.mean(b)
                b = copy.deepcopy(b1)
                input1[:, :, j] = copy.deepcopy(b)
            input[i, :, :, :] = copy.deepcopy(input1)
    else:
        input1 = copy.deepcopy(input)
        for j in range(s[-1]):
            b = copy.deepcopy(input1[:, :, j])
            b1 = b - np.mean(b)
            b = copy.deepcopy(b1)
            input1[:, :, j] = copy.deepcopy(b)
        input = copy.deepcopy(input1)
    return input


def preprocess(inputi):
    #input = copy.deepcopy(valid_images)
    s = inputi.shape
    if len(s) == 4:
        input = np.zeros([s[0], s[1], s[2], s[3]])
        # for i in range(s[0]):
        #     input1 = copy.deepcopy(input[i, :, :, :])
        #     for j in range(s[-1]):
        #         b = copy.deepcopy(input1[:, :, j])
        #         b1 = b - np.mean(b)
        #         b = copy.deepcopy(b1)
        #         input1[:, :, j] = copy.deepcopy(b)
        #     input[i, :, :, :] = copy.deepcopy(input1)
        ele = 1
        #imageURL = 'D:/redo3/images/training/1972.png'
        #b1 = misc.imread('D:/redo3/annotations/training/1972.png')
        #imageRGB = misc.imread(imageURL)
        He = np.array([0.644211, 0.716556, 0.266844], 'float64')
        Eo = np.array([0.092789, 0.954111, 0.283111], 'float64')
        Res = np.array([0, 0, 0], 'float64')
        for i in range(3):
            if He[i]**2+Eo[i]**2 > 1:
                Res[i] = 0.001
            else:
                Res[i] = np.sqrt(1-He[i]**2-Eo[i]**2)
        HDABtoRGB = np.matrix([He/np.linalg.norm(He), Eo/np.linalg.norm(Eo), Res/np.linalg.norm(Res)])
        RGBtoHDAB = np.linalg.inv(HDABtoRGB)
        for j in range(s[0]):
            k = 0
            k = k + 1
            input1 = copy.deepcopy(inputi[j, :, :, :])
            imageHDAB = SeparateStains(input1, RGBtoHDAB)
            #misc.imsave('D:\\'+str(k)+'.png', imageHDAB[:, :, 0])
            cc = np.zeros([s[1], s[2], s[3]])
            c = copy.deepcopy(imageHDAB[:, :, 0])
            if c.max() != c.min():
               c = np.round(255*(c-c.min())/(c.max()-c.min()))
            else:
               c = np.round(255 + c-c.min())
            c = copy.deepcopy(np.array(c, 'uint8'))
            for i in range(3):
               cc[:, :, i] = copy.deepcopy(c)
            #misc.imsave('E:\\1.png', cc)
            #cc1 = misc.imread('E:\\1.png')
            input[j, :, :, :] = copy.deepcopy(cc)
        # input = preprocess1(input)
    if len(s) == 3:
        # for i in range(s[0]):
        #     input1 = copy.deepcopy(input[i, :, :, :])
        #     for j in range(s[-1]):
        #         b = copy.deepcopy(input1[:, :, j])
        #         b1 = b - np.mean(b)
        #         b = copy.deepcopy(b1)
        #         input1[:, :, j] = copy.deepcopy(b)
        #     input[i, :, :, :] = copy.deepcopy(input1)
        ele = 1
        # imageURL = 'D:/redo3/images/training/1972.png'
        # b1 = misc.imread('D:/redo3/annotations/training/1972.png')
        # imageRGB = misc.imread(imageURL)
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
        input1 = copy.deepcopy(inputi)
        imageHDAB = SeparateStains(input1, RGBtoHDAB)
        # misc.imsave('D:\\'+str(k)+'.png', imageHDAB[:, :, 0])
        cc = np.zeros([s[0], s[1], s[2]])
        c = copy.deepcopy(imageHDAB[:, :, 0])
        if c.max() != c.min():
            c = np.round(255 * (c - c.min()) / (c.max() - c.min()))
        else:
            c = np.round(255 + c - c.min())
        c = copy.deepcopy(np.array(c, 'uint8'))
        for i in range(3):
            cc[:, :, i] = copy.deepcopy(c)
        # misc.imsave('E:\\1.png', cc)
        # cc1 = misc.imread('E:\\1.png')
        # input = preprocess1(cc)
    return input
