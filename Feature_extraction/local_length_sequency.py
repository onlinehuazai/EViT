import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor


def length_sequence_each_component(accof, dccof, row, col, type, N=8):
    _, acarr = jacdecColor(accof, type)
    _, dcarr = jdcdecColor(dccof, type, 'E')
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)
    retFeature = np.zeros((int(row*col/64), 64))
    blockIndex = 0
    Eob = np.where(acarr == 999)[0]
    count = 0  # 计算 Eob 位置
    dc_idx = 0
    ac_idx = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ac_idx: Eob[count]]
            ac_idx = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[dc_idx], ac)
            dc_idx = dc_idx + 1
            az = np.zeros(64 - acc.shape[0])
            acc = np.append(acc, az)

            # 对 acc计算长度
            coePosition = 0
            # 对非0系数提取位数信息
            for coefficient in acc:
                if coefficient == 0:
                    retFeature[blockIndex, coePosition] = 0
                else:
                    tmp = bin(int(coefficient))
                    coeLen = len(tmp)
                    if tmp[0] == '-':
                        coeLen -= 3
                    else:
                        coeLen -= 2
                    retFeature[blockIndex, coePosition] = coeLen
                coePosition += 1

            blockIndex += 1
    if type == 'Y':
        return retFeature
    else:
        return retFeature[:, :32]


def length_sequence_all_component(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size):
    featureAll = []
    featureY = length_sequence_each_component(acallY, dcallY, int(img_size[0]), int(img_size[1]), 'Y')
    featureCb = length_sequence_each_component(acallCb, dcallCb, int(img_size[0]), int(img_size[1]), 'Cb')
    featureCr = length_sequence_each_component(acallCr, dcallCr, int(img_size[0]), int(img_size[1]), 'Cr')
    featureAll.append(np.concatenate([featureY, featureCb, featureCr], axis=1).astype(np.int8))
    return featureAll
