from kaglib.PairGenerator import *
from kaglib.matchers.LoFTR import *
from kaglib.matchers.KeyNetAffNetHardNet import *
from kaglib.utils import read_csv_data_path, print_data_dict, create_submission
from kaglib.PycolmapHandler import *
import logging
import gc
import os.path as op
from collections import defaultdict
import pycolmap
import copy

src = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train'
device = torch.device('cuda')
cwd = op.dirname(__file__)
csv_path = op.join(
    cwd, 'input/image-matching-challenge-2023/train/train_labels_urban.csv')

data_dict = read_csv_data_path(csv_path)
out_results = defaultdict(dict)

for dataset, _ in data_dict.items():
    for scene in data_dict[dataset]:
        img_dir = f'{src}/{dataset}/{scene}/images'
        if not os.path.exists(img_dir):
            continue

        # Wrap the meaty part in a try-except block.
        out_results[dataset][scene] = {}
        img_fnames = [f'{src}/{x}' for x in data_dict[dataset][scene]]
        print(f"Got {len(img_fnames)} images")

        t = time()
        index_pairs = get_image_pairs_shortlist(
            img_fnames,
            sim_th=0.5,  # should be strict
            min_pairs=
            20,  # we select at least min_pairs PER IMAGE with biggest similarity
            exhaustive_if_less=20,
            device=device)
        t = time() - t
        print(index_pairs)
        print(f'{len(index_pairs)}, pairs to match, {t:.4f} sec')

        t = time()
        feature_dir = f'/home/jsmoon/kaggle/featureout/{dataset}_{scene}'
        if not os.path.isdir(feature_dir):
            os.makedirs(feature_dir, exist_ok=True)
        if LOCAL_FEATURE != 'LoFTR':
            detect_features(img_fnames,
                            2048,
                            feature_dir=feature_dir,
                            upright=True,
                            device=device,
                            resize_small_edge_to=600)
            t = time() - t
            print(f'Features detected in  {t:.4f} sec')
            t = time()
            match_features(img_fnames,
                           index_pairs,
                           feature_dir=feature_dir,
                           device=device)
        else:
            match_loftr(img_fnames,
                        index_pairs,
                        feature_dir=feature_dir,
                        device=device,
                        resize_to_=(600, 800))
        t = time() - t
        print(f'Features matched in  {t:.4f} sec')

        database_path = f'{feature_dir}/colmap.db'
        if os.path.isfile(database_path):
            os.remove(database_path)
        import_into_colmap(img_dir,
                           feature_dir=feature_dir,
                           database_path=database_path)
        output_path = f'{feature_dir}/colmap_rec_{LOCAL_FEATURE}'

        t = time()
        pycolmap.match_exhaustive(database_path)
        t = time() - t
        print(f'RANSAC in  {t:.4f} sec')

        t = time()
        # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options.min_model_size = 3
        os.makedirs(output_path, exist_ok=True)
        maps = pycolmap.incremental_mapping(database_path=database_path,
                                            image_path=img_dir,
                                            output_path=output_path,
                                            options=mapper_options)
        print(maps)
        #clear_output(wait=False)
        t = time() - t
        print(f'Reconstruction done in  {t:.4f} sec')
        imgs_registered = 0
        best_idx = None
        print("Looking for the best reconstruction")
        if isinstance(maps, dict):
            for idx1, rec in maps.items():
                print(idx1, rec.summary())
                if len(rec.images) > imgs_registered:
                    imgs_registered = len(rec.images)
                    best_idx = idx1
        if best_idx is not None:
            for k, im in maps[best_idx].images.items():
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
        print(
            f'Registered: {dataset} / {scene} -> {len(out_results[dataset][scene])} images'
        )
        print(
            f'Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images'
        )

create_submission(out_results, data_dict)