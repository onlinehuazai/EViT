import time
import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor


def exHuffman(dch, ach, bin_dch, bin_ach):
    hist_dct = np.histogram(dch, bins=bin_dch)
    hist_dct = hist_dct[0]
    # hist_dc = hist_dct/np.sum(hist_dct)
    hist_dc = hist_dct
    hist_act = np.histogram(ach, bins=bin_ach)
    hist_act = hist_act[0]
    # hist_ac = hist_act/np.sum(hist_act)
    hist_ac = hist_act
    return hist_dc, hist_ac


def global_feature(dccofY, accofY, dccofCb, accofCb, dccofCr, accofCr):
    allYdch = []
    allYach = []
    allCbdch = []
    allCbach = []
    allCrdch = []
    allCrach = []
    # start = time.time()
    Yach, _ = jacdecColor(accofY, 'Y')
    Ydch, _ = jdcdecColor(dccofY, 'Y')
    Cbach, _ = jacdecColor(accofCb, 'C')
    Cbdch, _ = jdcdecColor(dccofCb, 'C')
    Crach, _ = jacdecColor(accofCr, 'C')
    Crdch, _ = jdcdecColor(dccofCr, 'C')
    # end = time.time()
    # print(end - start)
    allYdch.append(np.array(Ydch).astype(np.int8))
    allYach.append(np.array(Yach).astype(np.int8))
    allCbdch.append(np.array(Cbdch).astype(np.int8))
    allCbach.append(np.array(Cbach).astype(np.int8))
    allCrach.append(np.array(Crach).astype(np.int8))
    allCrdch.append(np.array(Crdch).astype(np.int8))

    # DC/AC Huffman Table rows
    bin_ach = [i for i in range(0, 163)]
    bin_dch = [i for i in range(0, 13)]
    print('gloabal feature extraction')
    Ys = exHuffman(allYdch, allYach, bin_dch, bin_ach)
    allYdchist = Ys[0]
    allYachist = Ys[1]
    Cbs = exHuffman(allCbdch, allCbach, bin_dch, bin_ach)
    allCbdchist = Cbs[0]
    allCbachist = Cbs[1]
    Crs = exHuffman(allCrdch, allCrach, bin_dch, bin_ach)
    allCrdchist = Crs[0]
    allCrachist = Crs[1]
    golbal_feature = np.concatenate([allYdchist, allYachist, allCbdchist, allCbachist, allCrdchist, allCrachist], axis=0)
    return golbal_feature
