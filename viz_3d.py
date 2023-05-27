#%%
import pycolmap
from hloc.utils import viz_3d

src = '/home/jsmoon/kaggle/spsg/haiper_bike'
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
    '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train/haiper/bike/images'
)
visualization.visualize_sfm_2d(rec_gt, images, color_by='visibility', n=25)
# %%
