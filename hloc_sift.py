import tqdm, tqdm.notebook
import torch
import os
import os.path as op
import copy

from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.utils.io import get_keypoints, get_matches
from kaglib.utils import create_submission
from collections import defaultdict
from kaglib.utils import read_csv_data_path, create_submission
import pycolmap
from ensemble import merge_keypoints, merge_matches
import gc

src = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train'
device = torch.device('cuda:7')
cwd = op.dirname(__file__)
csv_path = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train/train_labels.csv'

feature_conf = {
    'output': 'feats-sift',
    'model': {
        'name': 'dog'
    },
    'preprocessing': {
        'grayscale': True,
        'resize_max': 2400,
    }
}
matcher_conf = {
    'output': 'matches-adalam',
    'num_workers': 0,
    'model': {
        'name': 'adalam'
    }
}

feature_confs = []
ensemble_sizes = [2400, 2800]
for ensemble_size in ensemble_sizes:
    feature_conf = copy.deepcopy(feature_conf)
    feature_conf['preprocessing']['resize_max'] = ensemble_size
    feature_confs.append(feature_conf)

data_dict = read_csv_data_path(csv_path)
out_results = defaultdict(dict)
print(data_dict.keys())

for dataset, _ in data_dict.items():
    for scene in data_dict[dataset]:
        img_dir = f'{src}/{dataset}/{scene}/images'
        if not os.path.exists(img_dir):
            continue
        out_results[dataset][scene] = {}
        images = Path(f'{src}/{dataset}/{scene}/images')

        features_list = []
        matches_list = []

        outputs = Path(f'{cwd}/sift/{dataset}_{scene}')
        if not os.path.isdir(outputs):
            os.makedirs(outputs, exist_ok=True)

        sfm_pairs = outputs / 'pairs-sfm.txt'
        sfm_dir = outputs / 'sfm'
        references = [str(p.relative_to(images)) for p in images.iterdir()]
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)

        for idx, feature_conf in enumerate(feature_confs):

            features = outputs / f'features_{idx}.h5'
            matches = outputs / f'matches_{idx}.h5'
            features_list.append(features)
            matches_list.append(matches)

            print(len(references), "mapping images")

            extract_features.main(feature_conf,
                                  images,
                                  image_list=references,
                                  feature_path=features)
            match_features.main(matcher_conf,
                                sfm_pairs,
                                features=features,
                                matches=matches)
            gc.collect()
            torch.cuda.empty_cache()

            print(f'ensemble {idx}/{len(feature_confs)} done')

        print('Merging features and matches...')
        merge_keypoints(features_list)
        merge_matches(matches_list, features_list, sfm_pairs)

        merged_features = outputs / 'merged_features.h5'
        merged_matches = outputs / 'merged_matches.h5'

        options = {'min_model_size': 3}
        model = reconstruction.main(sfm_dir,
                                    images,
                                    sfm_pairs,
                                    merged_features,
                                    merged_matches,
                                    image_list=references,
                                    mapper_options=options)
        if model:
            for k, im in model.images.items():
                key1 = f'{dataset}/{scene}/images/{im.name}'
                R = copy.deepcopy(im.rotmat())
                t = copy.deepcopy(im.tvec)
                # if R is None or t is None or not isinstance(
                #         R, np.ndarray) or not isinstance(t, np.ndarray):
                #     raise OSError(key1)
                out_results[dataset][scene][key1] = {}
                out_results[dataset][scene][key1]["R"] = copy.deepcopy(
                    im.rotmat())
                out_results[dataset][scene][key1]["t"] = copy.deepcopy(im.tvec)
                print("im print!!!!")
                print(im.rotmat())
                print(im.tvec)
create_submission(out_results, data_dict)