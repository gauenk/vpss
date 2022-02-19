
# -- python --
from easydict import EasyDict as edict

# -- linalg --
import numpy as np
import torch as th

# -- sim search import --
from .utils import optional
from .l2norm_impl import compute_l2norm_cuda
from .fill_patches import get_patches_burst
from .needle_impl import get_needles

def exec_sim_search(srch_img,srch_inds,sigma,k=None,
                    inds=None,vals=None,flows=None,**kwargs):
    """
    For each location from "srch_inds"
    we access the patches from each image
    and we return the "top K" image_indices
    indicating the center of a patch
    """

    # -- optionally no batch index  --
    no_batch = srch_img.ndim == 4
    if no_batch:
        srch_img = srch_img[None,:]
        srch_inds = srch_inds[None,:]

    # -- create output --
    device = srch_img.device
    B,T,C,H,W = srch_img.shape
    B,N,three = srch_inds.shape

    # -- optionally allocate space --
    if inds is None and k is None:
        raise ValueError("Can't have both [inds] and [k] be None.")
    if k is None:
        k = inds.shape[-1]
    elif inds is None:
        inds = th.zeros((B,N,k),dtype=th.long).to(device)
    else:
        assert inds.shape[2] == k
    if vals is None:
        vals = th.zeros((B,N,k),dtype=th.float32).to(device)
    if flows is None:
        flows = allocate_flows(T,H,W,device)
    assert inds.shape[-1] == k
    assert vals.shape[-1] == k

    # -- search each elem of batch --
    for b in range(B):
        exec_sim_search_burst(srch_img[b],srch_inds[b],vals[b],
                              inds[b],flows,sigma,kwargs)

    # -- optional flatten if no batch --
    if no_batch:
        inds = inds[0]

    return inds

def allocate_flows(T,H,W,device):
    flows = edict()
    zflow = th.zeros(T,2,H,W,dtype=th.float,device=device)
    flows.fflow = zflow
    flows.bflow = zflow
    return flows

def exec_sim_search_burst(srch_img,srch_inds,vals,inds,flows,sigma,args):
    stype = optional(args,"stype","l2")
    if stype == "l2":
        exec_sim_search_burst_l2(srch_img,srch_inds,vals,inds,flows,sigma,args)
    elif stype == "needle":
        exec_sim_search_burst_needle(srch_img,srch_inds,vals,inds,flows,sigma,args)
    else:
        raise NotImplemented("")

def exec_sim_search_burst_needle(srch_img,srch_inds,vals,inds,flows,sigma,args):

    # -- standard search --
    kneedle = 500
    N,k = vals.shape
    device = srch_img.device
    l2_vals = th.zeros((N,kneedle),dtype=th.float32).to(device)
    l2_inds = th.zeros((N,kneedle),dtype=th.long).to(device)
    exec_sim_search_burst_l2(srch_img,srch_inds,l2_vals,l2_inds,flows,sigma,args)

    # -- unpack options --
    pt = optional(args,'pt',2)
    pt = optional(args,'ps_t',pt)
    ps = optional(args,"ps",7)
    nps = optional(args,"nps",23)
    nscales = optional(args,'nscales',8)
    scale = optional(args,'needle_scale',0.75)

    # -- patches --
    patches = get_patches_burst(srch_img,l2_inds,ps,cs=None,pt=pt)

    # -- get needle --
    needles = get_needles(patches,nps,nscales,scale)

    # -- delta of top 500 by needles --
    mean_dims = (-5,-4,-3,-2,-1)
    n_vals = th.mean((needles[:,[0]] - needles)**2,mean_dims)

    # -- compute top k --
    device,b = n_vals.device,n_vals.shape[0]
    get_topk(n_vals,l2_inds,vals,inds)


def exec_sim_search_burst_l2(srch_img,srch_inds,vals,inds,flows,sigma,args):

    # -- fixed params --
    device = srch_img.device
    T,C,H,W = srch_img.shape
    step_s = 1 # this does nothing
    ps = optional(args,"ps",7)
    w_s = optional(args,'w_s',27)
    nWt_f = optional(args,'nWt_f',6)
    nWt_b = optional(args,'nWt_b',6)
    step = optional(args,'step',0)
    step1 = step == 0
    pt = optional(args,'pt',2)
    pt = optional(args,'ps_t',pt)
    cs_ptr = optional(args,'cs_ptr',th.cuda.default_stream().cuda_stream)
    offset = optional(args,'offset',0)

    # -- compute values for srch_inds --
    l2_vals,l2_inds = compute_l2norm_cuda(srch_img,flows.fflow,flows.bflow,
                                          srch_inds,step_s,ps,pt,w_s,nWt_f,
                                          nWt_b,step1,offset,cs_ptr)


    # -- compute top k --
    device,b = l2_vals.device,l2_vals.shape[0]
    get_topk(l2_vals,l2_inds,vals,inds)

def ensure_srch_inds(inds,srch_inds,h,w,c):
    """
    Ensure each search inds is at index 0 of each "inds"
    """
    print("ensure.")

    # -- compute index 0 --
    inds0 = srch_inds[:,0].clone() * h * w * c
    inds0 += srch_inds[:,1] * w
    inds0 += srch_inds[:,2]

    # -- get inds with "inds" already included somewhere --
    delta = th.abs(inds0[:,None] - inds)
    print("delta.shape: ",delta.shape)
    # delta = th.any(delta.transpose(1,0)<1e-8,1)
    print("delta.shape: ",delta.shape)
    incl_inds = th.nonzero(delta<1e-8)
    print(incl_inds)

    print("h,w,c: ",h,w,c)

    print(srch_inds[:3])
    print(inds[:3,:5])
    print(inds0[:3])
    print(incl_inds)

    # print(inds[0])

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

