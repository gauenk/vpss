
# -- python --
import math
from easydict import EasyDict as edict

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- separate package --
import hids

# -- sim search import --
from .utils import optional,get_3d_inds
from .l2norm_impl import compute_l2norm_cuda,compute_l2norm_cuda_fast
from .l2norm_impl import python_faiss_cuda
from .l2norm_impl import sim_cuda
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
    assert (srch_img.ndim == 5) and (srch_inds.ndim == 3)

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
    elif stype == "faiss":
        exec_sim_search_burst_faiss(srch_img,srch_inds,vals,inds,flows,sigma,args)
    elif stype == "sim_image":
        exec_sim_search_similar(srch_img,srch_inds,vals,inds,flows,args)
    elif stype == "needle":
        exec_sim_search_burst_needle(srch_img,srch_inds,vals,inds,flows,sigma,args)
    else:
        raise ValueError(f"stype = [{stype}]")

def exec_sim_search_burst_needle(srch_img,srch_inds,vals,inds,flows,sigma,args):

    # -- standard search --
    N,k = vals.shape
    kneedle = 200
    device = srch_img.device
    l2_vals = th.zeros((N,kneedle),dtype=th.float32).to(device)
    l2_inds = -th.ones((N,kneedle),dtype=th.long).to(device)
    exec_sim_search_burst_l2(srch_img,srch_inds,l2_vals,l2_inds,flows,sigma,args)

    # -- simple case --
    # exec_sim_search_burst_l2(srch_img,srch_inds,vals,inds,flows,sigma,args)
    # return

    # -- unpack options --
    pt = optional(args,'pt',2)
    pt = optional(args,'ps_t',pt)
    ps = optional(args,"ps",7)
    nps = optional(args,"nps",7)
    nscales = optional(args,'nscales',8)
    scale = optional(args,'needle_scale',0.75)
    ipos = optional(args,'ipos',"top-left")

    # -- patches --
    # we want the "ps" patch in the center of a larger patch --
    ci = 1 if args.step == 0 else 3
    scale = 0.75
    iscale = math.ceil(1/(scale**nscales))
    ips = iscale * nps
    ips += (ips % 2) == 0
    poff = (ips - ps)/2.
    assert poff == int(poff),"int valued for now."
    poff = int(poff)
    # poff = 0
    # ips = ps
    # print("ips: ",ips)
    # print("poff: ",poff)
    # print("ipos: ",ipos)
    # print("iscale: ",iscale)
    # print("srch_img[min,max]: ",srch_img.min().item(),srch_img.max().item())
    patches = get_patches_burst(srch_img,l2_inds,ips,cs=None,pt=pt,poff=poff)

    # -- get needle --
    # print("patches[min,max]: ",patches.min().item(),patches.max().item())
    needles = get_needles(patches,nps,nscales,scale)
    # needles = patches[:,:,None,:,:,:,:]
    # needles = patches[:,:,None,:,:,4:11,4:11]
    # needles = patches[:,:,None,:,:,12:19,12:19]
    # needles = needles[:,:,[0]]
    # print("[2] needles.shape: ",needles.shape)

    # -- beam search for subset --
    # pshape = th.IntTensor(list(needles.shape[2:]))
    # pshape = (pshape[0]*pshape[1],pshape[2],pshape[3],pshape[4])
    # needles = rearrange(needles[...,:ci,:,:],'b n s t c h w -> b n 1 (s t c h w)')
    # n_vals,n_inds = hids.subset_search(needles[:,:,0],sigma,k,"beam",
    #                                    bwidth=10,swidth=10,
    #                                    num_search = 10, max_mindex=3,
    #                                    svf_method="svar_needle",pshape=pshape)
    # inds[...] = th.gather(l2_inds,1,n_inds)
    # vals[...] = n_vals[...,None]

    # -- compute clusters --
    # clusters = cluster_needles(needles,sigma)

    # -- extract patch --

    #
    # -- select method v2 [simple] --
    #

    # -- delta of top 500 by needles --
    mean_dims = (-5,-4,-3,-2,-1)
    n_vals = th.mean((needles[:,[0],:,:,:ci] - needles[:,:,:,:,:ci])**2,mean_dims)
    n_vals[th.nonzero(l2_inds == -1,as_tuple=True)] = float("inf")

    # -- compute top k --
    device,b = n_vals.device,n_vals.shape[0]
    get_topk(n_vals,l2_inds,vals,inds)

def exec_sim_search_burst_l2_fast(srch_img,srch_inds,
                                  vals,inds,
                                  srch_dists,srch_locs,srch_bufs,
                                  flows,sigma,args):

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
    l2_vals,l2_inds = compute_l2norm_cuda_fast(srch_img,flows.fflow,flows.bflow,
                                               srch_dists,srch_locs,srch_bufs,
                                               srch_inds,step_s,ps,pt,w_s,nWt_f,
                                               nWt_b,step1,offset,cs_ptr)

    # -- compute top k --
    device,b = l2_vals.device,l2_vals.shape[0]
    get_topk(l2_vals,l2_inds,vals,inds)


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

def exec_sim_search_burst_faiss(srch_img,srch_inds,vals,inds,flows,sigma,args):

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
    l2_vals,l2_inds = python_faiss_cuda(srch_img,flows.fflow,flows.bflow,
                                        srch_inds,step_s,ps,pt,w_s,nWt_f,
                                        nWt_b,step1,offset,cs_ptr)

    # -- compute top k --
    device,b = l2_vals.device,l2_vals.shape[0]
    if l2_vals.shape[0] > 90000:
        th.cuda.empty_cache()
    get_topk(l2_vals,l2_inds,vals,inds)

def exec_sim_search_similar(noisy,access,vals,inds,flows,args):

    # -- unpack --
    ps = optional(args,'ps',7)
    pt = optional(args,'pt',1)
    ws = optional(args,'ws',27)
    wf = optional(args,'wf',6)
    wb = optional(args,'wb',6)
    k = optional(args,'k',2)
    chnls = optional(args,'chnls',1)
    ref = optional(args,'ref',-1)

    # -- flows --
    fflow = flows.fflow
    bflow = flows.bflow

    # -- exec search --
    l2_vals,l2_inds = sim_cuda(noisy,access,fflow,bflow,ref,ps,pt,chnls,ws,wf,wb,k)

    # -- run topk --
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


# --------------------------
#
#    Exhaustive Searching
#
# --------------------------

def exh_search_default_args():
    args = edict({'ps':7,'pt':1,'c':3})
    args['stype'] = "faiss"
    args['vpss_mode'] = "exh"
    args['bstride'] = 1
    return args

def exh_search_default(clean,flows,sigma,K):
    args = exh_search_default_args
    args.c = int(clean.shape[1])
    return exh_search(clean,flows,sigma,K,args)

def get_flows(flows,t,h,w,device):
    # -- handle flows --
    tf32 = th.float32
    if flows is None:
        flows = edict()
    if not("fflow" in flows):
        fshape = (t,2,h,w)
        flows.fflow = th.zeros(fshape,dtype=tf32,device=device)
        flows.bflow = flows.fflow
    return flows

def exh_search(clean,flows,sigma,K,args,clock=None):

    # -- unpack --
    device = clean.device
    shape = clean.shape
    t,c,h,w = shape

    # -- get flows --
    flows = get_flows(flows,t,h,w,device)

    # -- get search inds --
    index,BSIZE,stride = 0,t*h*w,args.bstride
    srch_inds = get_search_inds(index,BSIZE,stride,shape,device)
    srch_inds = srch_inds.type(th.int32)
    # print("srch_inds.shape: ",srch_inds.shape)

    # -- get return shells --
    numQueries = ((BSIZE - 1)//args.bstride)+1
    # print("numQueries: ",numQueries,args.bstride,BSIZE)
    nq,pt,c,ps = numQueries,args.pt,args.c,args.ps
    vals,inds,patches = init_topk_shells(nq,K,pt,c,ps,device)

    # -- search using numba code --
    clock.tic()
    exec_sim_search_burst(clean,srch_inds,vals,inds,flows,sigma,args)

    # -- fill patches --
    get_patches_burst(clean,inds,ps,pt=pt,patches=patches,fill_mode="faiss")

    # -- weight floating-point issue --
    patches = patches.type(th.float32)
    th.cuda.synchronize()
    clock.toc()

    return vals,inds,patches


def init_topk_shells(bsize,k,pt,c,ps,device):
    tf32,ti32 = th.float32,th.int32
    vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
    inds = -th.ones((bsize,k),dtype=ti32,device=device)
    patches = -th.ones((bsize,k,pt,c,ps,ps),dtype=tf32,device=device)
    return vals,inds,patches

def get_search_inds(index,bsize,stride,shape,device):
    t,c,h,w  = shape
    start = index * bsize
    stop = ( index + 1 ) * bsize
    ti32 = th.int32
    srch_inds = th.arange(start,stop,stride,dtype=ti32,device=device)[:,None]
    srch_inds = get_3d_inds(srch_inds,h,w)
    srch_inds = srch_inds.contiguous()
    return srch_inds

