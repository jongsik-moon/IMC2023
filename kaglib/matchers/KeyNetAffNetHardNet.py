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

LOCAL_FEATURE = 'LoFTR'


class KeyNetAffNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
            self,
            num_features: int = 5000,
            upright: bool = False,
            device=torch.device('cpu'),
            scale_laf: float = 1.0,
    ):
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(
            angle_detector=KF.OriNet(False)).eval()
        if not upright:
            weights = torch.load(
                '/home/jsmoon/kaggle/input/kornia-local-feature-weights/OriNet.pth'
            )['state_dict']
            ori_module.angle_detector.load_state_dict(weights)
        detector = KF.KeyNetDetector(
            False,
            num_features=num_features,
            ori_module=ori_module,
            aff_module=KF.LAFAffNetShapeEstimator(False).eval()).to(device)
        kn_weights = torch.load(
            '/home/jsmoon/kaggle/input/kornia-local-feature-weights/keynet_pytorch.pth'
        )['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load(
            '/home/jsmoon/kaggle/input/kornia-local-feature-weights/AffNet.pth'
        )['state_dict']
        detector.aff.load_state_dict(affnet_weights)

        hardnet = KF.HardNet(False).eval()
        hn_weights = torch.load(
            '/home/jsmoon/kaggle/input/kornia-local-feature-weights/HardNetLib.pth'
        )['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = KF.LAFDescriptor(hardnet,
                                      patch_size=32,
                                      grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)


def detect_features(img_fnames,
                    num_feats=2048,
                    upright=False,
                    device=torch.device('cpu'),
                    feature_dir='.featureout',
                    resize_small_edge_to=600):
    if LOCAL_FEATURE == 'DISK':
        # Load DISK from Kaggle models so it can run when the notebook is offline.
        disk = KF.DISK().to(device)
        pretrained_dict = torch.load(
            '/kaggle/input/disk/pytorch/depth-supervision/1/loftr_outdoor.ckpt',
            map_location=device)
        disk.load_state_dict(pretrained_dict['extractor'])
        disk.eval()
    if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
        feature = KeyNetAffNetHardNet(num_feats, upright,
                                      device).to(device).eval()
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
         h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in progress_bar(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                timg = load_torch_image(img_path, device=device)
                H, W = timg.shape[2:]
                if resize_small_edge_to is None:
                    timg_resized = timg
                else:
                    timg_resized = K.geometry.resize(timg,
                                                     resize_small_edge_to,
                                                     antialias=True)
                    print(
                        f'Resized {timg.shape} to {timg_resized.shape} (resize_small_edge_to={resize_small_edge_to})'
                    )
                h, w = timg_resized.shape[2:]
                if LOCAL_FEATURE == 'DISK':
                    features = disk(timg_resized,
                                    num_feats,
                                    pad_if_not_divisible=True)[0]
                    kps1, descs = features.keypoints, features.descriptors

                    lafs = KF.laf_from_center_scale_ori(
                        kps1[None],
                        torch.ones(1, len(kps1), 1, 1, device=device))
                if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
                    lafs, resps, descs = feature(
                        K.color.rgb_to_grayscale(timg_resized))
                lafs[:, :, 0, :] *= float(W) / float(w)
                lafs[:, :, 1, :] *= float(H) / float(h)
                desc_dim = descs.shape[-1]
                kpts = KF.get_laf_center(lafs).reshape(
                    -1, 2).detach().cpu().numpy()
                descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                f_laf[key] = lafs.detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return



def match_features(img_fnames,
                   index_pairs,
                   feature_dir='.featureout',
                   device=torch.device('cpu'),
                   min_matches=15,
                   force_mutual=True,
                   matching_alg='smnn'):
    assert matching_alg in ['smnn', 'adalam']
    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:

        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
            lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            if matching_alg == 'adalam':
                img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
                hw1, hw2 = img1.shape[:2], img2.shape[:2]
                adalam_config = KF.adalam.get_adalam_default_config()
                #adalam_config['orientation_difference_threshold'] = None
                #adalam_config['scale_rate_threshold'] = None
                adalam_config['force_seed_mnn'] = False
                adalam_config['search_expansion'] = 16
                adalam_config['ransac_iters'] = 128
                adalam_config['device'] = device
                dists, idxs = KF.match_adalam(
                    desc1,
                    desc2,
                    lafs1,
                    lafs2,  # Adalam takes into account also geometric information
                    hw1=hw1,
                    hw2=hw2,
                    config=adalam_config
                )  # Adalam also benefits from knowing image size
            else:
                dists, idxs = KF.match_smnn(desc1, desc2, 0.98)
            if len(idxs) == 0:
                continue
            # Force mutual nearest neighbors
            if force_mutual:
                first_indices = get_unique_idxs(idxs[:, 1])
                idxs = idxs[first_indices]
                dists = dists[first_indices]
            n_matches = len(idxs)
            if False:
                print(f'{key1}-{key2}: {n_matches} matches')
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2,
                                     data=idxs.detach().cpu().numpy().reshape(
                                         -1, 2))
    return

