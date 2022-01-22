# -- linalg --
import numpy as np
import torch as th

# -- vision --
import torchvision.transforms as tvt

# -- package import --
from .sobel import apply_sobel_filter

# -- local imports --
from .impl import exec_sim_search
from .fill_patches import fill_patches

def imgs2patches(noisy,clean,sigma,ps,npatches,nneigh,ps_l2=None,pt=2):

    # -- hyper-params --
    c = noisy.shape[-3]

    # -- select candidate inds --
    srch_inds = select_patch_inds(clean,npatches,ps)

    # -- return neighbors of "search indices" via l2 search --
    ps_l2 = ps if ps_l2 is None else ps_l2
    inds = exec_sim_search(noisy,srch_inds,nneigh,sigma,ps_l2)

    # -- create patches from inds --
    pnoisy = construct_patches(noisy,inds,ps,pt)
    pclean = construct_patches(clean,inds,ps,pt)

    return pnoisy,pclean


def select_patch_inds(img,npatches,ps):


    # -- shapes and init --
    device = img.device
    B,T,C,H,W = img.shape
    inds = th.zeros((B,npatches,3),dtype=th.long,device=device)

    # -- for each elem ... --
    for b in range(B):

        # -- compute edges --
        edges = apply_sobel_filter(img[b])

        # -- don't grab near edges --
        edges[...,:32,:32] = -100
        edges[...,-32:,-32:] = -100

        # -- sample indices prop. to edge weight --
        perc95 = th.quantile(edges[...,32:-32,32:-32].ravel(),0.95)
        mask = edges > perc95
        index = th.nonzero(mask)
        order = th.randperm(index.shape[0])
        inds[b,...] = index[order[:npatches]]

    return inds

def construct_patches(img,inds,ps,pt):

    # -- init patches --
    device = inds.device
    B,N,k = inds.shape
    B,T,c,H,W = img.shape
    patches = th.zeros((B,N,k,pt,c,ps,ps),dtype=th.float,device=device)

    # -- parameter --
    cs_ptr = th.cuda.default_stream().cuda_stream

    # -- fill each batch --
    for b in range(B):
        fill_patches(patches[b],img[b],inds[b],cs_ptr)

    return patches
