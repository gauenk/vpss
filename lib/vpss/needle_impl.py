"""

The "Needle-Match" distance between two
patches using the CVPR 2016 paper
Thank you Lotan & Irani :D

"""

# -- python --
import sys,pdb
import numpy as np
import torch as th
from einops import rearrange,repeat


import torchvision
from torch.nn.functional import grid_sample

# -- numba --
from numba import jit,njit,prange,cuda

# -- this package --
from .l2norm_impl import compute_l2norm_cuda
from .fill_patches import get_patches_burst

# def compute_needle_cuda(noisy,fflow,bflow,access,nps,ps,ps_t,w_s,
#                         nWt_f,nWt_b,step1,offset,cs,k=1):

#     # -- standard matching for top 500 --
#     dists,inds = compute_l2norm_cuda(noisy,fflow,bflow,access,None,ps,ps_t,w_s,
#                                      nWt_f,nWt_b,step1,offset,cs,k=1)
#     patches = get_patches_burst(noisy,inds,ps,cs=None,pt=ps_t)

#     # -- default --
#     nscales = 8
#     needles = get_needles(patches,ps,nps,nscales,scale)
#     needles = get_needles(patches,needle_ps)
#     return needles

def get_inds(ps):
    inds = []
    psHalf,m = ps//2,ps%2
    for i in range(-psHalf,psHalf+m):
        for j in range(-psHalf,psHalf+m):
            coord_ij = [i,j]
            inds.append(coord_ij)
    inds = th.FloatTensor(inds)
    return inds

def get_grid(ps,h,w):
    inds = get_inds(ps)
    inds[:,0] = 2*(inds[:,0]/h - 0.5)
    inds[:,1] = 2*(inds[:,1]/w - 0.5)
    inds = inds.reshape(1,ps,ps,2)
    return inds.double()

def get_needles(patches,nps,nscales,scale):

    # -- info --
    no_batch = patches.ndim == 6
    if no_batch: patches = patches[None,:]

    # -- create grid --
    device = patches.device
    B,N,K,pt,c,ph,pw = patches.shape
    grid = get_grid(nps,ph,pw).to(device).float()

    # -- re-shaping --
    patches = rearrange(patches,'b n k t c h w -> (b n k t) c h w')
    R = patches.shape[0]
    grid = grid.expand(R,-1,-1,-1)

    # -- create needle --
    needles = []
    for n in range(nscales):
        scale_s = scale**n
        grid_s = grid / scale_s
        patch_s = grid_sample(patches,grid_s,mode="bicubic",padding_mode="reflection",
                              align_corners=True)
        needles.append(patch_s)
    needles = th.stack(needles)
    print("needles.shape: ",needles.shape)

    # -- reshaping --
    needles = rearrange(needles,'s (b n k t) c h w -> b n k s t c h w',b=B,n=N,k=K)
    print("needles.shape: ",needles.shape)

    return needles
