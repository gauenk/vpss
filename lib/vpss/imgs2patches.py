
# -- linalg --
import numpy as np
import torch as th

# -- vision --
import torchvision.transforms as tvt

# -- package import --
from .utils import optional
from .sobel import apply_sobel_filter

# -- local import --
from .fill_patches import fill_patches
from .impl import exec_sim_search

def imgs2patches(noisy,clean,sigma,ps,npatches,nneigh,**kwargs):
    """

    Converts each image to a set of "npatches" each with "nneigh" patches

    """

    # -- hyper-params --
    pt = optional(kwargs,'pt',2)
    c = optional(kwargs,'c',3)

    # -- select candidate inds --
    srch_inds = select_patch_inds(clean,npatches,ps)

    # -- return neighbors of "search indices" via l2 search --
    inds = exec_sim_search(noisy,srch_inds,sigma,k=nneigh,ps=ps,pt=pt)
    assert th.any(inds==-1).item() is False, "must have no invalid srch_inds"

    # -- create patches from inds --
    pnoisy = construct_patches(noisy,inds,ps,pt)
    pclean = construct_patches(clean,inds,ps,pt)

    return pnoisy,pclean,inds


def select_patch_inds(img,npatches,ps):


    # -- shapes and init --
    device = img.device
    B,T,C,H,W = img.shape
    inds = th.zeros((B,npatches,3),dtype=th.long,device=device)
    BDR = 8

    # -- for each elem ... --
    for b in range(B):

        # -- compute edges --
        edges = apply_sobel_filter(img[b])

        # -- don't grab near edges --
        edges[...,:BDR,:] = -100
        edges[...,:,:BDR] = -100
        edges[...,:,-BDR:] = -100
        edges[...,-BDR:,:] = -100

        # -- sample indices prop. to edge weight --
        perc95 = th.quantile(edges[...,BDR:-BDR,BDR:-BDR].ravel(),0.95)
        mask = edges > perc95

        # -- filter out edges --
        mask[...,:BDR,:] = 0
        mask[...,:,:BDR] = 0
        mask[...,-BDR:,:] = 0
        mask[...,:,-BDR:] = 0
        mask[T-1,...] = 0

        # -- select inds --
        index = th.nonzero(mask)
        order = th.randperm(index.shape[0])
        inds[b,...] = index[order[:npatches]]

    return inds

def construct_patches(img,inds,ps,pt):

    # -- init patches --
    device = inds.device
    B,N,k = inds.shape
    B,T,c,H,W = img.shape

    # -- allocate memo --
    patches = th.zeros((B,N,k,pt,c,ps,ps),dtype=th.float,device=device)

    # -- parameter --
    cs_ptr = th.cuda.default_stream().cuda_stream

    # -- fill each batch --
    for b in range(B):
        fill_patches(patches[b],img[b],inds[b],cs_ptr)

    return patches
