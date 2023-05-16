# General utilities
import os
from tqdm import tqdm
from time import time
import numpy as np

# CV/ML
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def get_global_desc(fnames, model, device=torch.device('cpu')):
    model = model.eval()
    model = model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    global_descs_convnext = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        img = Image.open(img_fname_full).convert('RGB')
        timg = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            desc = model.forward_features(timg.to(device)).mean(dim=(-1, 2))  #
            #print (desc.shape)
            desc = desc.view(1, -1)
            desc_norm = F.normalize(desc, dim=1, p=2)
        #print (desc_norm)
        global_descs_convnext.append(desc_norm.detach().cpu())
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i + 1, len(img_fnames)):
            index_pairs.append((i, j))
    return index_pairs


def get_image_pairs_shortlist(
    fnames,
    sim_th=0.3,  # should be strict
    min_pairs=20,
    exhaustive_if_less=20,
    device=torch.device('cpu')):
    num_imgs = len(fnames)

    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)

    model = timm.create_model(
        'tf_efficientnet_b7',
        checkpoint_path=
        '/home/jsmoon/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b7/1/tf_efficientnet_b7_ra-6c08e654.pth'
    )
    model.eval()
    descs = get_global_desc(fnames, model, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs - 1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total += 1
    matching_list = sorted(list(set(matching_list)))
    return matching_list