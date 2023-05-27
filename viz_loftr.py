#%%
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from pathlib import Path

from hloc import extract_features, visualization, pairs_from_exhaustive, match_dense, pairs_from_retrieval
from hloc.visualization import plot_images, read_image
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import parse_retrieval

src = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train'
dataset = 'haiper'
scene = 'bike'
sfm = f'/home/jsmoon/kaggle/loftr/{dataset}_{scene}'
images = Path(f'{src}/{dataset}/{scene}/images')
outputs = Path(f'/home/jsmoon/kaggle/query/{dataset}_{scene}')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

model = pycolmap.Reconstruction(f'{sfm}/sfm')

references = [p.relative_to(images).as_posix() for p in (images).iterdir()]
ref_ids = [r for r in model.images]
queries = [img.name for _, img in model.images.items()]
query = queries[1]
print(query)

matcher_conf = match_dense.confs['loftr']
retrieval_conf = extract_features.confs['netvlad']

model = pycolmap.Reconstruction(f'{sfm}/sfm')

references = [p.relative_to(images).as_posix() for p in (images).iterdir()]
ref_ids = [r for r in model.images]
queries = [img.name for _, img in model.images.items()]
query = queries[1]
print(query)

pairs_from_exhaustive.main(sfm_pairs, image_list=[query], ref_list=references)
features, sfm_matches = match_dense.main(matcher_conf,
                                            sfm_pairs,
                                            images,
                                            outputs,
                                            max_kps=8192,
                                            overwrite=True)

camera = pycolmap.infer_camera_from_image(images / query)
conf = {
    'estimation': {
        'ransac': {
            'max_error': 12
        }
    },
    'refinement': {
        'refine_focal_length': True,
        'refine_extra_params': True
    },
}
localizer = QueryLocalizer(model, conf)
ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features,
                             sfm_matches)

print(
    f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.'
)
visualization.visualize_loc_from_log(images, query, log, model, top_k_db=5)
# %%
