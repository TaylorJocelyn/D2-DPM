import numpy as np
import random
# import shuffle
import shutil
import copy
import os
import path

def shuffle_img():
    file_list = []

    # for file in os.listdir('/home/zq/PTQD/evaluate_data/scale3.0_eta0.0_steps20/fp32'):
    #     file_list.append(file)

    # num = len(file_list)
    # idx = [x for x in range(num)]
    # random.shuffle(idx)

    # idx = 0
    # for cfile in file_list:
    #     src_path = os.path.join('/home/zq/PTQD/evaluate_data/scale3.0_eta0.0_steps20/fp32', cfile)
    #     dst_path = '/home/zq/PTQD/evaluate_data/scale3.0_eta0.0_steps20/fp32_random/' + f'img_{idx: 05d}.jpg'
    #     idx += 1

    #     shutil.copy(src_path, dst_path)

    path = '/home/zq/PTQD/evaluate_data/scale3.0_eta0.0_steps20/fp32'
    for file in path.glob('*.{}'.format(jpg)):
        print(file)


if __name__ == '__main__':
    shuffle_img()