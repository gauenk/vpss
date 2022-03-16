

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- local --
from .impl import exec_sim_search

def compute_warp(ref,img2warp):

    # -- unpack --
    device = ref.device
    c,h,w = img2warp.shape
    DIM = h*w*c

    # -- inits --
    warp = th.zeros_like(img2warp)
    burst = th.stack([ref,img2warp])

    # -- batching --
    bsize = 1024
    nbatches = (h*w - 1) // (bsize+1) # div UP
    bstart = 0

    # -- range of batching --
    for batch in range(nbatches):

        # -- create search inds --
        srch_inds = th.arange(bstart,bstart+bsize).reshape(1,-1)
        srch_inds = get_3d_inds(srch_inds,c,h,w).to(device)
        bstart += bsize

        # -- find nn --
        kwargs = {'nWt_b':0,'nWt_f':1,'w_s':27,'ps':7,'step':1,"pt":1}
        inds = exec_sim_search(burst,srch_inds,0.,k=54,**kwargs)
        print("inds.shape: ",inds.shape)

        # -- get first "t+1" index; e.g. where in ref are we? --
        print(inds[0])
        print(inds,DIM)
        inds -= DIM
        args = th.where(inds >= 0)[0]
        print(args)
        inds = inds[th.where(inds >= DIM)]
        assert len(inds) > 0


        # -- fill at locs --
        for ci in range(c):
            warp[ci,...] = th.gather(img2warp[ci].ravel(),inds,0)

    return warp


def get_3d_inds(inds,c,h,w):

    # -- unpack --
    chw,hw = c*h*w,h*w
    bsize,num = inds.shape
    device = inds.device

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill --
    aug_inds[0,...] = tdiv(inds,chw,rounding_mode='floor') # inds // chw
    aug_inds[1,...] = tdiv(tmod(inds,hw),w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    return aug_inds


