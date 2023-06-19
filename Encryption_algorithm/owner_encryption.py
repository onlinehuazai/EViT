## laod plain-images and secret keys
import numpy as np
import scipy.io as scio
from encryption_utils import ksa
from encryption_utils import prga
from encryption_utils import yates_shuffle
import tqdm
from encryption_utils import loadImageSet, loadImageFiles
from JPEG.rgbandycbcr import rgb2ycbcr
import cv2
import copy
from JPEG.jdcencColor import jdcencColor
from JPEG.zigzag import zigzag
from JPEG.invzigzag import invzigzag
from JPEG.jacencColor import jacencColor
from JPEG.Quantization import *
from cipherimageRgbGenerate import Gen_cipher_images
import multiprocessing as mul
import datetime
import glob


def encryption_each_component(image_component, keys, p_key, type, row, col, N, QF):
    allblock8 = np.zeros([8, 8, int(row * col / (8 * 8))])
    allblock8_number = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            t = image_component[m:m + N, n:n + N] - 128
            allblock8[:, :, allblock8_number] = t
            allblock8_number = allblock8_number + 1

    # block shuffle
    block8_number = int((row * col) / (8 * 8))
    data = [i for i in range(0, block8_number)]
    p_block = yates_shuffle(data, p_key)
    allblock8_permute = copy.copy(allblock8)
    for i in range(0, len(p_block)):
        allblock8_permute[:, :, i] = allblock8[:, :, p_block[i]-1]
    allblock8 = copy.copy(allblock8_permute)
    del allblock8_permute
    # Huffman coding
    dc = 0
    dccof= []
    accof = []
    for i in range(0, allblock8_number):
        t = copy.copy(allblock8[:, :, i])
        t = cv2.dct(t)  # DCT
        temp = Quantization(t, type=type)  # Quanlity
        if i == 0:
            dc = temp[0, 0]
            key_numbers, dc_component = jdcencColor(dc, type, keys)
            dccof = np.append(dccof, dc_component)
            keys = keys[key_numbers:]
        else:
            dc = temp[0, 0] - dc
            key_numbers, dc_component = jdcencColor(dc, type, keys)
            dccof = np.append(dccof, dc_component)
            dc = temp[0, 0]
            keys = keys[key_numbers:]
        acseq = []
        aczigzag = zigzag(temp)
        eobi = 0
        for j in range(63, -1, -1):
            if aczigzag[j] != 0:
                eobi = j
                break
        if eobi == 0:
            acseq = np.append(acseq, [999])
        else:
            acseq = np.append(acseq, aczigzag[1: eobi + 1])
            acseq = np.append(acseq, [999])
        key_numbers, ac_component = jacencColor(acseq, type, keys)
        keys = keys[key_numbers:]
        accof = np.append(accof, ac_component)

    return dccof, accof


def get_size(img):
    row, col, _ = img.shape

    # if row > col:
    #     img = cv2.resize(img, (128, 192))
    # else:
    #     img = cv2.resize(img, (192, 128))

    row, col, _ = img.shape
    plainimage = rgb2ycbcr(img)
    plainimage = plainimage.astype(np.float16)

    Y = plainimage[:, :, 0]
    Cb = plainimage[:, :, 1]
    Cr = plainimage[:, :, 2]

    for i in range(0, int(8 * np.ceil(col / 8) - col)):
        Y = np.c_[Y, Y[:, -1]]
        Cb = np.c_[Cb, Cb[:, -1]]
        Cr = np.c_[Cr, Cr[:, -1]]

    for i in range(0, int(8 * np.ceil(row / 8) - row)):
        Y = np.r_[Y, [Y[-1, :]]]
        Cb = np.r_[Cb, [Cb[-1, :]]]
        Cr = np.r_[Cr, [Cr[-1, :]]]
    return 8 * np.ceil(row / 8), 8 * np.ceil(col / 8), Y, Cb, Cr



def encryption(Y, Cb, Cr, keyY, keyCb, keyCr, p_keyY, p_keyCb, p_keyCr, QF, N, row, col):
    # N: block size
    # QF: quality factor

    row = int(row)
    col = int(col)
    # Cb = cv2.resize(Cb,
    #                 (int(col / 2), int(row / 2)),
    #                 interpolation=cv2.INTER_CUBIC)
    # Cr = cv2.resize(Cr,
    #                 (int(col / 2), int(row / 2)),
    #                 interpolation=cv2.INTER_CUBIC)

    # Y component
    dccofY, accofY = encryption_each_component(Y, keyY, p_keyY, type='Y', row=row, col=col, N=N, QF=QF)
    ## Cb and Cr component
    dccofCb, accofCb = encryption_each_component(Cb, keyCb, p_keyCb, type='Cb', row=int(row), col=int(col),
                                                 N=N, QF=QF)
    dccofCr, accofCr = encryption_each_component(Cr, keyCr, p_keyCr, type='Cr', row=int(row), col=int(col),
                                                 N=N, QF=QF)
    accofY = accofY.astype(np.int8)
    dccofY = dccofY.astype(np.int8)
    accofCb = accofCb.astype(np.int8)
    dccofCb = dccofCb.astype(np.int8)
    accofCr = accofCr.astype(np.int8)
    dccofCr = dccofCr.astype(np.int8)
    return accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr



def main(imageFile):
    # read plain-image
    # if imageFile.split('\\')[-1].split('.')[0] not in bitFiles_pre:
    img = cv2.imread(imageFile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (256, 192))
    row, col, Y, Cb, Cr = get_size(img)
    # if row * col > max_key_len:
    # encryption_keyY, encryption_keyCb, encryption_keyCr = generate_keys(row * col)

    accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr = encryption(Y, Cb, Cr, encryption_keyY,
                                                                    encryption_keyCb,
                                                                    encryption_keyCr,
                                                                    p_keyY, p_keyCb, p_keyCr,
                                                                    QF,
                                                                    N=8, row=row, col=col)

    # if not os.path.exists(bit_path + imageFile.split('\\')[-2]):
    #     os.mkdir(bit_path + imageFile.split('\\')[-2])
    img_size = (row, col)
    np.save(bit_path + imageFile.split('\\')[-1].split('.')[0] + '.npy',
            {'accofY': accofY, 'dccofY': dccofY, 'accofCb': accofCb, 'dccofCb': dccofCb, 'accofCr': accofCr,
             'dccofCr': dccofCr, 'size': img_size})
    print(imageFile + 'process success!')
    # Gen_cipher_images(dccofY, accofY, dccofCb, accofCb, dccofCr, accofCr, img_size, imageFile)


QF = 100
max_key_len = 256*384
# load keys
#####################
imageFiles = loadImageFiles('../data/plainimages/*.jpg')
bit_path = '../data/JPEGBitStream/'
# read bitstream
bitFiles = glob.glob(bit_path+'/*.npy')

if __name__ == '__main__':
    # image encryption
    now_time = datetime.datetime.now()
    print(now_time)
    pool = mul.Pool(3)
    rel = pool.map(main, imageFiles)
    now_time = datetime.datetime.now()
    print(now_time)
