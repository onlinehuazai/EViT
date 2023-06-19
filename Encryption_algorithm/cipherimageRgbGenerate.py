## generate cipherimage
import numpy as np
from JPEG.jacdecColorHuffman import jacdecColor
from JPEG.jdcdecColorHuffman import jdcdecColor
from JPEG.invzigzag import invzigzag
import cv2
from JPEG.rgbandycbcr import ycbcr2rgb, rgb2ycbcr
from JPEG.DCT import idctJPEG
from JPEG.Quantization import iQuantization
import os


def deEntropy(acall, dcall, row, col, type, N=8, QF = 100):
    _, acarr = jacdecColor(acall, type)
    _, dcarr = jdcdecColor(dcall, type)
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)
    row = int(row)
    col = int(col)
    Eob = np.where(acarr == 999)
    Eob = Eob[0]
    count = 0
    kk = 0
    ind1 = 0
    xq = np.zeros([row, col])
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ind1: Eob[count]]
            ind1 = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[kk], ac)
            az = np.zeros(64 - acc.shape[0])
            acc = np.append(acc, az)
            temp = invzigzag(acc, 8, 8)
            temp = iQuantization(temp, QF, type)
            temp = idctJPEG(temp)
            xq[m:m + N, n:n + N] = temp + 128
            kk = kk + 1
    return xq


def Gen_cipher_images(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size, path):
    cipher_Y = deEntropy(acallY, dcallY, img_size[0], img_size[1], 'Y')
    cipher_cb = deEntropy(acallCb, dcallCb, img_size[0], img_size[1], 'U')
    cipher_cr = deEntropy(acallCr, dcallCr, img_size[0], img_size[1], 'V')
    cipherimage = np.dstack([cipher_Y, cipher_cb, cipher_cr])
    # if not os.path.exists('../data/cipherimages/{}'.format(path.split('\\')[-2])):
    #     os.mkdir('../data/cipherimages/{}'.format(path.split('\\')[-2]))
    cipherimage = np.round(cipherimage)
    cipherimage = cipherimage.astype(np.uint8)
    cipherimage = ycbcr2rgb(cipherimage)
    merged = cv2.merge([cipherimage[:, :, 2], cipherimage[:, :, 1], cipherimage[:, :, 0]])
    cv2.imwrite('../data/cipherimages/{}'.format(path.split("\\")[-1]), merged, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('pictures completed')
