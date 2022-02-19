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

from pathlib import Path


import torchvision
import torchvision.utils as tvu
from torch.nn.functional import grid_sample

# -- numba --
from numba import jit,njit,prange,cuda

# -- this package --
from .l2norm_impl import compute_l2norm_cuda
from .fill_patches import get_patches_burst
from .utils import yuv2rgb_patches,apply_yuv2rgb

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

def get_grid(ps,h,w,ipos):
    """

    "r" is the ratio we "squeeze" our
    initial grid from the first grid.

    without "r" our first grid spans the
    entire intial (h,w) image

    but we want the first patch
    to be the (ps,ps) inset image of
    the greater (h,w) image

    """
    inds = get_inds(ps)
    if ipos == "center":
        inds[:,0] = inds[:,0]/h*2
        inds[:,1] = inds[:,1]/w*2
    else:
        raise ValueError(f"Uknown index position [ipos = {ipos}]")

    inds = inds.reshape(1,ps,ps,2)
    return inds.double()

def get_needles(patches,nps,nscales,scale,ipos="center"):

    # -- info --
    no_batch = patches.ndim == 6
    if no_batch: patches = patches[None,:]

    # -- create grid --
    device = patches.device
    B,N,K,pt,c,ph,pw = patches.shape
    grid = get_grid(nps,ph,pw,ipos).to(device).float()

    # -- re-shaping --
    patches = rearrange(patches,'b n k t c h w -> (b n k t) c h w')
    R = patches.shape[0]
    grid = grid.expand(R,-1,-1,-1)

    # -- create needle --
    needles = []
    for n in range(nscales):

        # -- re-scale --
        scale_s = scale**n
        # -- simple, centered growth --
        grid_s = update_grid(grid,scale_s,ipos)
        # print("grid_s[min,max]: ",grid_s.min().item(),grid_s.max().item())
        patch_s = grid_sample(patches,grid_s,mode="bicubic",
                              padding_mode="reflection",
                              align_corners=False)
        # patch_s = patches.clone()[...,9:16,9:16]
        needles.append(patch_s)
    needles = th.stack(needles)

    # -- save examples --
    save_ex = False
    if save_ex:
        nsave = 10
        save_dir = Path("./output/needle/")
        if not save_dir.exists(): save_dir.mkdir()
        # yuv2rgb_patches
        print("patches.shape: ",patches.shape)
        print("needles.shape: ",needles.shape)
        save_patches(patches,save_dir,nsave)
        save_needles(needles,save_dir,nsave)
        exit(0)

    # -- reshaping --
    needles = rearrange(needles,'s (b n k t) c h w -> b n k s t c h w',b=B,n=N,k=K)
    # print("needles.shape: ",needles.shape)
    if no_batch: needles = needles[0]

    return needles


def update_grid(grid,scale_s,ipos):
    if ipos == "center":
        grid_s = grid / scale_s
    else:
        raise ValueError(f"Uknown index position [ipos = {ipos}]")
    return grid_s

def pgroup2rgb(pgroup):
    n,b,c,h,w = pgroup.shape
    pgroup = rearrange(pgroup,'n b c h w -> (n b) c h w')
    pgroup_rgb = pgroup.clone()
    apply_yuv2rgb(pgroup_rgb)
    pgroup_rgb = rearrange(pgroup_rgb,'(n b) c h w -> n b c h w',n=n)
    # pgroup = rearrange(pgroup,'n b c h w -> (b c) n (h w)')
    # pgroup_rgb = yuv2rgb_patches(pgroup.clone(),c=3,pt=1)
    # pgroup_rgb = rearrange(pgroup_rgb,'b n (c h w) -> n b c h w',c=c,h=h,w=w)
    return pgroup_rgb


#
# --- io --
#

def save_needles(needles,save_dir,nsave):
    soff = 100
    fn = str(save_dir / "needle.png")
    nneedles = needles.shape[0]
    # print("needles[min,max]: ",needles.min().item(),needles.max().item())
    sneedles = pgroup2rgb(needles)
    sneedles = rearrange(sneedles[:,soff:soff+nsave],'b k c h w -> (k b) c h w')/255.
    tvu.save_image(sneedles,fn,nrow=nneedles,value_range=[0.,1.])

def save_patches(patches,save_dir,nsave):
    offset = 100
    for i in range(500):
        fn = str(save_dir / ("patches_%d.png" % i))
        # print("patches[min,max]: ",patches.min().item(),patches.max().item())
        spatches = pgroup2rgb(patches[None,:])[0]
        spatches = spatches[10*i:10*(i+1)]/255.
        # print("spatches.mean(): ",spatches.mean())
        tvu.save_image(spatches,fn,value_range=[0.,1.])
