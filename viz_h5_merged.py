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


PATH_TO_FEATS = '/home/jsmoon/kaggle/sift/heritage_wall'
matches = load_h5_2layer(os.path.join(PATH_TO_FEATS, 'merged_matches.h5'))
features = load_h5_2layer(os.path.join(PATH_TO_FEATS, 'merged_features.h5'))
IMG_DIR = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train/heritage/wall/images'
img1_key = 'DSC_4928_acr.jpg'
img2_key = 'DSC_4940_acr.jpg'
img1 = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, img1_key)),
                    cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, img2_key)),
                    cv2.COLOR_BGR2RGB)
plt.imshow(np.concatenate([img1, img2], axis=1))

import kornia.feature as KF
import torch
from kornia_moons.feature import *

kpt0 = features[img1_key]['keypoints']
kpt1 = features[img2_key]['keypoints']
try:
    matches0 = matches[f'{img1_key}_{img2_key}']['matches0']
except:
    matches0 = matches[f'{img2_key}_{img1_key}']['matches0']
match_indices = np.where(matches0 != -1)[0]

# Filter keypoints and matches using match_indices
kpt0_matched = kpt0[match_indices]
kpt1_matched = kpt1[matches0[match_indices]]

# Convert matched keypoints to cv2.KeyPoint objects
keypoints0_cv2 = [
    cv2.KeyPoint(point[0], point[1], 1) for point in kpt0_matched
]
keypoints1_cv2 = [
    cv2.KeyPoint(point[0], point[1], 1) for point in kpt1_matched
]

# Create matches
matches_cv2 = [cv2.DMatch(i, i, 0) for i in range(len(match_indices))]
# Convert matches to cv2.DMatch objects

img_out = cv2.drawMatches(img1,
                          keypoints0_cv2,
                          img2,
                          keypoints1_cv2,
                          matches_cv2,
                          None,
                          flags=2)
plt.figure()
fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(img_out, interpolation='nearest')
cv2.imwrite('/home/jsmoon/result.png', img_out)
# %%
