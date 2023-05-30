#%%
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from pathlib import Path

from hloc import extract_features, match_features, visualization, pairs_from_exhaustive, reconstruction
from hloc.visualization import plot_images, read_image

src = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train'
dataset = 'haiper'
scene = 'chairs'
sfm = f'/home/jsmoon/kaggle/sosnet/{dataset}_{scene}'
images = Path(f'{src}/{dataset}/{scene}/images')
outputs = Path(f'/home/jsmoon/kaggle/sosnet/{dataset}_{scene}')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

model = pycolmap.Reconstruction(f'{sfm}/sfm')

references = [p.relative_to(images).as_posix() for p in (images).iterdir()]
queries = [img.name for _, img in model.images.items()]
print(queries)
query = queries[0]
print(query)
ref_ids = [id for id, img in model.images.items() if img.name != query]
print(ref_ids)

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
                             matches)

print(
    f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.'
)
visualization.visualize_loc_from_log(images, query, log, model, top_k_db=10)
# %%
