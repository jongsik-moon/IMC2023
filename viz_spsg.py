#%%
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from pathlib import Path

from hloc import extract_features, match_features, visualization, pairs_from_exhaustive, reconstruction
from hloc.visualization import plot_images, read_image

src = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train'
dataset = 'haiper'
scene = 'chairs'
sfm = f'/home/jsmoon/kaggle/spsg/{dataset}_{scene}'
images = Path(f'{src}/{dataset}/{scene}/images')
outputs = Path(f'/home/jsmoon/kaggle/spsg/{dataset}_{scene}')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

model = pycolmap.Reconstruction(f'{sfm}/sfm')

references = [p.relative_to(images).as_posix() for p in (images).iterdir()]
ref_ids = [r for r in model.images]
queries = [img.name for _, img in model.images.items()]
query = queries[1]
print(query)

extract_features.main(feature_conf,
                      images,
                      image_list=[query],
                      feature_path=features,
                      overwrite=False)
pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
match_features.main(matcher_conf,
                    loc_pairs,
                    features=features,
                    matches=matches,
                    overwrite=False)

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
visualization.visualize_loc_from_log(images, query, log, model, top_k_db=5)
# %%