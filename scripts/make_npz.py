import numpy as np
import torch
import os
from PIL import Image

def concat_npz():
    arr = np.zeros((50000, 256, 256, 3), dtype=np.uint8)

    with np.load('/home/zq/PTQD/evaluate_data/lsun_bedroom_eta1.0_step200/linear_correct_25000_steps200_eta1.0_typebedroom_1.npz') as data:
        arr1 = data['arr_0']

    with np.load('/home/zq/PTQD/evaluate_data/lsun_bedroom_eta1.0_step200/linear_correct_25000_steps200_eta1.0_typebedroom_2.npz') as data:
        arr2 = data['arr_0']

    arr[:25000] = arr1[:]
    arr[25000:] = arr2[:]

    np.savez('/home/zq/PTQD/evaluate_data/lsun_bedroom_eta1.0_step200/linear_correct_bedroom_w8a8.npz', arr)

def concat_npz2():
    arr = np.zeros((50000, 256, 256, 3), dtype=np.uint8)

    with np.load('/home/zq/PTQD/evaluate_data/scale3.0_eta0.0_steps20/gaussian_correct_25000_steps20_eta0.0_scale3.0_1.npz') as data:
        arr1 = data['arr_0']

    with np.load('/home/zq/PTQD/evaluate_data/scale3.0_eta0.0_steps20/gaussian_correct_12500_steps20_eta0.0_scale3.0_2.npz') as data:
        arr2 = data['arr_0']

    with np.load('/home/zq/PTQD/evaluate_data/scale3.0_eta0.0_steps20/gaussian_correct_12500_steps20_eta0.0_scale3.0_3.npz') as data:
        arr3 = data['arr_0']

    arr[:25000] = arr1[:]
    arr[25000:37500] = arr2[:]
    arr[37500:] = arr3[:]

    np.savez('evaluate_data/scale3.0_eta0.0_steps20/w8a8_gaussian_correct.npz', arr)

def npz_to_img():
    
    with np.load('/home/zq/PTQD/evaluate_data/scale1.5_eta1.0_steps250/cwi_all_w4a8.npz') as data:
        arr = data['arr_0']

    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i])
        img.save('/home/zq/PTQD/evaluate_data/scale1.5_eta1.0_steps250/cwi_all/' + f'img_{i:05d}.png')


def img_to_npz():

    arr_mk = np.zeros((50000, 256, 256, 3), dtype=np.uint8)

    idx = 0
    for file in os.listdir('/home/zq/PTQD/evaluate_data/scale1.5_eta1.0_steps250/cwi_all/'):
        img_path = '/home/zq/PTQD/evaluate_data/scale1.5_eta1.0_steps250/cwi_all/' + file
        image = Image.open(img_path)
        img_arr = np.asarray(image)

        arr_mk[idx] = img_arr
        print('idx ', idx)
        idx += 1

    np.savez('/home/zq/PTQD/evaluate_data/scale1.5_eta1.0_steps250/cwi_all_mk.npz', arr_mk)
        

def img_to_npz2():
    arr_mk = np.zeros((25000, 256, 256, 3), dtype=np.uint8)
    
    for i in range(25000):
        img_path = 'evaluate_data/scale3.0_eta1.0_steps20/ptq4dm_w4a8/' + f'img_{i:05d}.png'
        image = Image.open(img_path)
        img_arr = np.asarray(image)

        print('idx ', i)
        arr_mk[idx] = img_arr

    np.savez('evaluate_data/scale3.0_eta1.0_steps20/ptq4dm_w4a8_mk.npz', arr_mk)



if __name__ == '__main__':
    img_to_npz()
