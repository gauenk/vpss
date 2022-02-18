
# -- python imports --
import torch as th
from easydict import EasyDict as edict

# -- this package --
import vpss

# -- params --
device = 'cuda:0'
sigma = 30./255.
npatches = 10
ps = 7

# -- load images --
imgs = edict()
imgs.clean = th.rand(1,5,3,128,128).to(device)
imgs.noisy = imgs.clean + sigma*th.randn_like(imgs.clean)

# -- search params --
srch_img = imgs.noisy
srch_inds = vpss.select_patch_inds(imgs.clean,npatches,ps)

# -- run search --
inds = vpss.exec_sim_search(srch_img,srch_inds,sigma,k=10,stype="needle")
print(inds)

# # -- fill patches --
# for key in imgs.patch_images:

#     # -- skip --
#     pass_key = (imgs[key] is None) or (patches[key] is None)
#     if pass_key: continue

#     # -- fill --
#     vpss.fill_patches(patches[key],imgs[key],bufs.inds)
