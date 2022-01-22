
# -- linalg --
import numpy as np
import torch as th

# -- sim search import --
from .utils import optional
from .l2norm_impl import compute_l2norm_cuda

def exec_sim_search(srch_img,srch_inds,k,sigma,ps,**kwargs):

    # -- create output --
    device = srch_img.device
    B,T,C,H,W = srch_img.shape
    B,N,three = srch_inds.shape
    inds = th.zeros((B,N,k),dtype=th.long,device=device)

    # -- search each elem of batch --
    for b in range(B):
        inds_b = exec_l2_search_burst(srch_img[b],srch_inds[b],k,sigma,ps,**kwargs)
        inds[b,...] = inds_b

    return inds

def exec_l2_search_burst(srch_img,srch_inds,k,sigma,ps,**kwargs):

    # -- fixed params --
    device = srch_img.device
    T,C,H,W = srch_img.shape
    fflow = optional(kwargs,'fflow',th.zeros(T,C,H,W,dtype=th.float,device=device))
    bflow =  optional(kwargs,'bflow',fflow)
    step_s = 1 # this does nothing
    w_s = optional(kwargs,'w_s',27)
    nWt_f = optional(kwargs,'nWt_f',6)
    nWt_b = optional(kwargs,'nWt_b',6)
    step1 = optional(kwargs,'step1',0)
    pt = optional(kwargs,'pt',2)
    pt = optional(kwargs,'ps_t',pt)
    cs_ptr = th.cuda.default_stream().cuda_stream
    offset = 2*(sigma**2)


    # -- compute values for srch_inds --
    l2_vals,l2_inds = compute_l2norm_cuda(srch_img,bflow,bflow,srch_inds,step_s,ps,
                                          pt,w_s,nWt_f,nWt_b,step1,offset,cs_ptr)

    # -- compute top k --
    inds = get_topk_pair(l2_vals,l2_inds,k)[1]

    return inds


def get_topk_pair(vals_srch,inds_srch,k):
    device,b = vals_srch.device,vals_srch.shape[0]
    vals = th.FloatTensor(b,k).to(device)
    inds = th.IntTensor(b,k).to(device)
    get_topk(vals_srch,inds_srch,vals,inds)
    return vals,inds

def get_topk(l2_vals,l2_inds,vals,inds):

    # -- shape info --
    b,_ = l2_vals.shape
    _,k = vals.shape

    # -- take mins --
    # order = th.topk(-l2_vals,k,dim=1).indices
    order = th.argsort(l2_vals,dim=1,descending=False)
    # -- get top k --
    vals[:b,:] = th.gather(l2_vals,1,order[:,:k])
    inds[:b,:] = th.gather(l2_inds,1,order[:,:k])

