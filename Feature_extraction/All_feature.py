import glob
from global_huffman_code_frequency import global_feature
from local_length_sequency import length_sequence_all_component
from Encryption_algorithm.encryption_utils import loadEncBit
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mul
import datetime



def main(path):
    bitstream = loadEncBit(path).item()  # load encrypted bitstream
    local_feature = length_sequence_all_component(bitstream['dccofY'], bitstream['accofY'], bitstream['dccofCb'],
                                                  bitstream['accofCb'], bitstream['dccofCr'], bitstream['accofCr'], bitstream['size'])
    np.save("../data/features/difffeature_matrix/" + path.split('\\')[-1].split('.')[0] + ".npy", local_feature)
    global_Huffman_feature = global_feature(bitstream['dccofY'], bitstream['accofY'], bitstream['dccofCb'],
                                            bitstream['accofCb'], bitstream['dccofCr'], bitstream['accofCr'])
    np.save("../data/features/huffman_feature/" + path.split('\\')[-1].split('.')[0] + ".npy", global_Huffman_feature)


if __name__ == '__main__':
    bit_path = '../data/JPEGBitStream/*.npy'
    bitFiles = glob.glob(bit_path)
    pool = mul.Pool(4)
    rel = pool.map(main, bitFiles)
    print('完成')

