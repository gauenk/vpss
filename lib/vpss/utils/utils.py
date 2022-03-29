
# -- linalg --
import torch as th
from einops import rearrange,repeat


def optional(pydict,key,default):
    if pydict is None: return default
    if key in pydict: return pydict[key]
    elif hasattr(pydict,key): return getattr(pydict,key)
    else: return default

def get_3d_inds(inds,h,w,unpack=False):

    # -- unpack --
    hw = h*w # no "chw" in this code-base; its silly.
    bsize,num = inds.shape
    device = inds.device

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill --
    aug_inds[0,...] = tdiv(inds,hw,rounding_mode='floor') # inds // chw
    aug_inds[1,...] = tdiv(tmod(inds,hw),w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')
    if unpack:
        aug_inds = rearrange(aug_inds,'(b n) three -> b n three',b=bsize)

    return aug_inds



