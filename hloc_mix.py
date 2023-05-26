import tqdm, tqdm.notebook
import torch
import os
import os.path as op
import copy

from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, match_dense, pairs_from_retrieval, pairs_from_covisibility, triangulation
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.utils.parsers import parse_retrieval
from kaglib.utils import create_submission
from collections import defaultdict
from kaglib.utils import read_csv_data_path, create_submission
import pycolmap
from hloc.utils.io import list_h5_names

src = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train'
device = torch.device('cuda')
cwd = op.dirname(__file__)
csv_path = op.join(
    cwd, 'input/image-matching-challenge-2023/train/train_labels.csv')
num_loc = 5

data_dict = read_csv_data_path(csv_path)
out_results = defaultdict(dict)
print(data_dict)
for dataset, _ in data_dict.items():
    for scene in data_dict[dataset]:
        img_dir = f'{src}/{dataset}/{scene}/images'
        if not os.path.exists(img_dir):
            continue
        # if scene != 'cyprus':
        #     continue
        # Wrap the meaty part in a try-except block.
        out_results[dataset][scene] = {}

        images = Path(f'{src}/{dataset}/{scene}/images')
        outputs = Path(f'/home/jsmoon/kaggle/mix/{dataset}_{scene}')
        if not os.path.isdir(outputs):
            os.makedirs(outputs, exist_ok=True)
        sfm_pairs = outputs / 'pairs-sfm.txt'
        loc_pairs = outputs / 'pairs-loc.txt'
        sfm_dir = outputs / 'sfm'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'

        references = [str(p.relative_to(images)) for p in images.iterdir()]
        print(len(references), "mapping images")

        feature_conf = extract_features.confs['superpoint_aachen']
        matcher_conf = match_dense.confs['loftr_superpoint']

        features_sp = extract_features.main(feature_conf,
                                            images,
                                            image_list=references,
                                            feature_path=features)

        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        features, sfm_matches = match_dense.main(matcher_conf,
                                                 sfm_pairs,
                                                 images,
                                                 export_dir=outputs,
                                                 features_ref=features_sp)

        print(sfm_matches)
        options = {'min_model_size': 3}
        model = reconstruction.main(sfm_dir,
                                    images,
                                    sfm_pairs,
                                    features_sp,
                                    sfm_matches,
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
