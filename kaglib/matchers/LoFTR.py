# General utilities
import os
from tqdm import tqdm
from time import time
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy

# CV/ML
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from kaglib.utils import load_torch_image, get_unique_idxs


class LoFTR:

    def __init__(self, device, model_path):
        self.matcher = KF.LoFTR(pretrained=None)
        self.matcher.load_state_dict(torch.load(model_path)['state_dict'])
        self.matcher = self.matcher.to(device).eval()

    def make_loft_result_h5(self,
                            img_fnames,
                            index_pairs,
                            feature_dir='.featureout_loftr',
                            device=torch.device('cpu'),
                            min_matches=15,
                            resize_to_=(640, 480)):
        f_match = h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w')
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            # Load img1
            timg1 = K.color.rgb_to_grayscale(
                load_torch_image(fname1, device=device))
            H1, W1 = timg1.shape[2:]
            if H1 < W1:
                resize_to = resize_to_[1], resize_to_[0]
            else:
                resize_to = resize_to_
            timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
            h1, w1 = timg_resized1.shape[2:]

            # Load img2
            timg2 = K.color.rgb_to_grayscale(
                load_torch_image(fname2, device=device))
            H2, W2 = timg2.shape[2:]
            if H2 < W2:
                resize_to2 = resize_to[1], resize_to[0]
            else:
                resize_to2 = resize_to_
            timg_resized2 = K.geometry.resize(timg2,
                                              resize_to2,
                                              antialias=True)
            h2, w2 = timg_resized2.shape[2:]
            with torch.inference_mode():
                input_dict = {"image0": timg_resized1, "image1": timg_resized2}
                correspondences = self.matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            score = correspondences['confidence'].cpu().numpy()

            mkpts0[:, 0] *= float(W1) / float(w1)
            mkpts0[:, 1] *= float(H1) / float(h1)

            mkpts1[:, 0] *= float(W2) / float(w2)
            mkpts1[:, 1] *= float(H2) / float(h2)

            n_matches = len(mkpts1)
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2,
                                     data=np.concatenate([mkpts0, mkpts1],
                                                         axis=1))

    def make_unique_keypoints_matches(self, feature_dir):
        # Let's find unique loftr pixels and group them together.
        kpts = defaultdict(list)
        match_indexes = defaultdict(dict)
        total_kpts = defaultdict(int)
        with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f_match:
            for k1 in f_match.keys():
                group = f_match[k1]
                for k2 in group.keys():
                    matches = group[k2][...]
                    total_kpts[k1]
                    kpts[k1].append(matches[:, :2])
                    kpts[k2].append(matches[:, 2:])
                    current_match = torch.arange(len(matches)).reshape(
                        -1, 1).repeat(1, 2)
                    current_match[:, 0] += total_kpts[k1]
                    current_match[:, 1] += total_kpts[k2]
                    total_kpts[k1] += len(matches)
                    total_kpts[k2] += len(matches)
                    match_indexes[k1][k2] = current_match

        for k in kpts.keys():
            kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
        unique_kpts = {}
        unique_match_idxs = {}
        out_match = defaultdict(dict)
        for k in kpts.keys():
            uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(
                kpts[k]),
                                                       dim=0,
                                                       return_inverse=True)
            unique_match_idxs[k] = uniq_reverse_idxs
            unique_kpts[k] = uniq_kps.numpy()
        for k1, group in match_indexes.items():
            for k2, m in group.items():
                m2 = deepcopy(m)
                m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
                m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
                mkpts = np.concatenate([
                    unique_kpts[k1][m2[:, 0]],
                    unique_kpts[k2][m2[:, 1]],
                ],
                                       axis=1)
                unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts),
                                                      dim=0)
                m2_semiclean = m2[unique_idxs_current]
                unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0],
                                                       dim=0)
                m2_semiclean = m2_semiclean[unique_idxs_current1]
                unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1],
                                                       dim=0)
                m2_semiclean2 = m2_semiclean[unique_idxs_current2]
                out_match[k1][k2] = m2_semiclean2.numpy()
        with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
            for k, kpts1 in unique_kpts.items():
                f_kp[k] = kpts1

        with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
            for k1, gr in out_match.items():
                group = f_match.require_group(k1)
                for k2, match in gr.items():
                    group[k2] = match
        return


def match_loftr(images,
                pairs,
                feature_dir='.featureout_loftr',
                device=torch.device('cpu'),
                min_matches=100,
                resize_max=1024,
                max_keypoints=500):
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(
        torch.load(
            '/home/jsmoon/kaggle/input/loftr/pytorch/outdoor/1/loftr_outdoor.ckpt'
        )['state_dict'])
    matcher = matcher.to(device).eval()

    # First we do pairwise matching, and then extract "keypoints" from loftr matches.
    with h5py.File(f'{feature_dir}/loftr_temp.h5', mode='w') as f_match:
        for pair in progress_bar(pairs):
            key1, key2 = pair
            # Load img1
            timg1 = K.color.rgb_to_grayscale(
                load_torch_image(str(images / key1), device=device))

            # Get the current width and height
            H1, W1 = timg1.shape[2:]

            # Calculate the aspect ratio
            aspect_ratio = W1 / H1

            # Determine the new dimensions based on the resize_max and aspect_ratio
            if W1 > H1:
                new_width = resize_max
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = resize_max
                new_width = int(new_height * aspect_ratio)

            timg_resized1 = K.geometry.resize(timg1, (new_width, new_height),
                                              antialias=True)
            h1, w1 = timg_resized1.shape[2:]

            # Load img2
            timg2 = K.color.rgb_to_grayscale(
                load_torch_image(str(images / key2), device=device))
            # Get the current width and height
            H2, W2 = timg2.shape[2:]

            # Calculate the aspect ratio
            aspect_ratio = W2 / H2

            # Determine the new dimensions based on the resize_max and aspect_ratio
            if W2 > H2:
                new_width = resize_max
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = resize_max
                new_width = int(new_height * aspect_ratio)

            timg_resized2 = K.geometry.resize(timg2, (new_width, new_height),
                                              antialias=True)

            h2, w2 = timg_resized2.shape[2:]
            with torch.no_grad():
                input_dict = {"image0": timg_resized1, "image1": timg_resized2}
                correspondences = matcher(input_dict)
                score = correspondences['confidence'].cpu().numpy()

            sort_indices = np.argsort(score)
            if len(sort_indices) > max_keypoints:
                sort_indices = sort_indices[-max_keypoints:]

            mkpts0 = correspondences['keypoints0'][sort_indices].cpu().numpy()
            mkpts0[:, 0] *= float(W1) / float(w1)
            mkpts0[:, 1] *= float(H1) / float(h1)

            mkpts1 = correspondences['keypoints1'][sort_indices].cpu().numpy()
            mkpts1[:, 0] *= float(W2) / float(w2)
            mkpts1[:, 1] *= float(H2) / float(h2)

            n_matches = len(mkpts1)
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2,
                                     data=np.concatenate([mkpts0, mkpts1],
                                                         axis=1))
            del mkpts0, mkpts1, correspondences
            torch.cuda.empty_cache()
    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)
    with h5py.File(f'{feature_dir}/loftr_temp.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1,
                                                                   1).repeat(
                                                                       1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),
                                                   dim=0,
                                                   return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        uniq_kps_len = len(unique_kpts[k1])
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            mkpts = np.concatenate([
                unique_kpts[k1][m2[:, 0]],
                unique_kpts[k2][m2[:, 1]],
            ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts),
                                                  dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            matches = m2_semiclean2.numpy()

            match_targets = [-1 for i in range(uniq_kps_len)]
            for match in matches:
                match_targets[match[0]] = match[1]
            out_match[k1][k2] = match_targets

    with h5py.File(f'{feature_dir}/features_loftr.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            print(k)
            f_kp.create_group(k)
            f_kp[k].create_dataset('keypoints', data=kpts1)

    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group.create_group(k2)
                group[k2].create_dataset('matches0', data=match)
    return
