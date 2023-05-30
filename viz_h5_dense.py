#%%
import h5py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from collections import defaultdict


def load_h5_2layer(filename):
    '''Loads dictionary from hdf5 file'''

    dict_to_load = defaultdict(dict)
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key1 in keys:
            for key2 in f[key1].keys():
                dict_to_load[key1][key2] = f[key1][key2][()]
    return dict_to_load


def load_h5_3layer(filename):
    '''Loads dictionary from hdf5 file'''

    dict_to_load = defaultdict(dict)
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key1 in keys:
            dict_to_load[key1] = defaultdict(dict)
            for key2 in f[key1].keys():
                for key3 in f[key1][key2].keys():
                    dict_to_load[key1][key2][key3] = f[key1][key2][key3][()]
    return dict_to_load


PATH_TO_FEATS = '/home/jsmoon/kaggle/loftr/haiper_fountain'
matches = load_h5_3layer(
    os.path.join(PATH_TO_FEATS, 'matches-loftr_pairs-sfm.h5'))
IMG_DIR = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train/haiper/fountain/images'
img1_key = 'image_000.jpeg'
img2_key = 'image_007.jpeg'
img1 = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, img1_key)),
                    cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, img2_key)),
                    cv2.COLOR_BGR2RGB)
plt.imshow(np.concatenate([img1, img2], axis=1))

# %%
import kornia.feature as KF
import torch
from kornia_moons.feature import *

kpt0 = matches[img1_key][img2_key]['keypoints0']
kpt1 = matches[img1_key][img2_key]['keypoints1']
matches0 = matches[img1_key][img2_key]['matches0']

keypoints0_cv2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kpt0]
keypoints1_cv2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kpt1]

# Convert matches to cv2.DMatch objects
matches_cv2 = [cv2.DMatch(i, i, 0) for i in matches0]

img_out = cv2.drawMatches(img1, keypoints0_cv2, img2, keypoints1_cv2,
                          matches_cv2, None)
plt.figure()
fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(img_out, interpolation='nearest')

# %%
