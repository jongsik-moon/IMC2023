#%%
import pycolmap
from hloc.utils import viz_3d

src = '/home/jsmoon/kaggle/featureout/heritage_cyprus'
rec_gt = pycolmap.Reconstruction(f'{src}/sfm')

fig = viz_3d.init_figure()
# viz_3d.plot_cameras(fig, rec_gt, color='rgba(50,255,50, 0.5)', name="Ground Truth", size=10)
viz_3d.plot_reconstruction(fig,
                           rec_gt,
                           cameras=False,
                           color='rgba(201,56,110,0.5)',
                           name="Ground Truth",
                           cs=5)
fig.show()
# %%
from hloc import visualization
from pathlib import Path

images = Path(
    '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train/heritage/cyprus/images'
)
visualization.visualize_sfm_2d(rec_gt, images, color_by='visibility', n=25)

# %%
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

query = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train/heritage/cyprus/images/DSC_6480.JPG'
references = [p.relative_to(images).as_posix() for p in (images).iterdir()]
print(references)
outputs = Path(f'/home/jsmoon/kaggle/featureout/heritage_cyprus')

features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

camera = pycolmap.infer_camera_from_image(images / query)
print(rec_gt.find_image_with_name('DSC_6480.JPG'))
ref_ids = [rec_gt.find_image_with_name(r).image_id for r in references]
conf = {
    'estimation': {'ransac': {'max_error': 12}},
    'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
}
localizer = QueryLocalizer(rec_gt, conf)
ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
visualization.visualize_loc_from_log(images, query, log, rec_gt)
# %%
