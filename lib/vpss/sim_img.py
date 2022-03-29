
# -- python --
import math
from easydict import EasyDict as edict

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- local imports --
from .impl import exec_sim_search_burst,exec_sim_search_similar
from .fill_patches import get_patches_burst
from .fill_image import fill_sim_image

# -- sim search import --
from . import utils
from .utils import optional,get_3d_inds


#
# Interface Function
#

def get_sim_from_burst(burst,ref,**kwargs):

    # -- get arguments --
    args = get_sim_args(**kwargs)

    # -- burst 2 device --
    burst = burst.to(args.device)

    # -- get flows --
    flows = get_flows(burst.shape,**kwargs)

    # -- build sim frame --
    sim_img = get_sim_image(burst,ref,flows,args)

    return sim_img

#
# Pair of images is a TYPE of burst
#

def get_sim_from_pair(src,tgt,**kwargs):

    # -- convert tensors from numpy --
    if not th.is_tensor(src):
        src = th.from_numpy(src)
    if not th.is_tensor(tgt):
        tgt = th.from_numpy(tgt)

    # -- create burst --
    burst = th.stack([src,tgt])

    return get_sim_from_burst(burst,0,**kwargs)


#
# Core Nearest-Neighbor Search Logic
#

def get_sim_image(burst,ref,flows,args):

    # -- batching info --
    t,c,h,w = burst.shape
    shape = burst.shape
    nelems,nsearch = get_batching_info(args,shape)
    start,stop = get_search_limits(shape,ref)
    args.ref = ref

    # -- allocs --
    patches = alloc_patches(args,args.bsize)
    inds = alloc_nn_buf(args.bsize,args.k,th.int32,args.device)
    vals = alloc_nn_buf(args.bsize,args.k,th.float32,args.device)
    sim_image = th.zeros(*shape[1:],dtype=th.uint8,device=args.device)

    # -- color xform --
    # srch_burst = rgb2yuv(burst)
    srch_burst = burst
    args.chnls = 3

    # -- exec search --
    for index in range(nsearch):

        # -- get search inds --
        srch_inds = get_search_inds(index,start,stop,shape,args)
        bsize = srch_inds.shape[0]

        # -- get views --
        patches_view = view_batch(patches,0,bsize)
        vals_view = view_batch(vals,0,bsize)
        inds_view = view_batch(inds,0,bsize)

        # -- search using numba code --
        exec_sim_search_similar(srch_burst,srch_inds,vals_view,inds_view,flows,args)

        # -- fill images --
        inds3d = get_3d_inds(inds_view,h,w,True)
        # ninds = th.where(inds3d[:,0,0] != ref)[0]
        # for i in range(min(100,len(ninds))):
        #     print(i,ninds[i],inds_view[ninds[i]],vals_view[ninds[i]])
        #     print(i,ninds[i],inds3d[ninds[i]],inds_view[ninds[i]],srch_inds[ninds[i]])
        assert th.all(inds3d[:,0,0] == ref)
        assert th.all(inds3d[:,1,0] == 1)

        # inds3d[:,1,0] = 1
        # inds3d[:,1,1] = inds3d[:,0,1]
        # inds3d[:,1,2] = inds3d[:,0,2]
        fill_sim_image(sim_image,burst,inds3d,args)

    return sim_image


# ---------------------------------------
#
#           Helper Functions
#
# ---------------------------------------

#
# Batching Help
#

def get_batching_info(args,shape):
    t,c,h,w = shape
    npix = h*w
    nelems = ((npix-1) // args.stride) + 1
    nsearch = ((nelems-1) // args.bsize) + 1
    return nelems,nsearch

def get_search_limits(shape,ref):
    t,c,h,w = shape
    start = ref * h * w
    stop = (ref+1) * h * w
    return start,stop

def view_batch(tensor,index,bsize):
    start = index * bsize
    stop = (index+1) * bsize
    return tensor[start:stop]

#
# Tensor Allocation
#

def alloc_patches(args,nelems):
    device = args.device
    ps,pt,c = args.ps,args.pt,args.c
    c,k = args.c,args.k
    pshape = (nelems,k,pt,c,ps,ps)
    patches = th.zeros(pshape,device=device,dtype=th.float32)
    return patches

def alloc_nn_buf(nelems,k,dtype,device):
    tshape = (nelems,k)
    tensor = th.zeros(tshape,device=device,dtype=dtype)
    return tensor

#
# Burst Formatting
#

def rgb2yuv(burst):
    images = edict()
    images.burst = burst.clone()
    images.ikeys = ['burst']
    utils.rgb2yuv_images(images)
    burst = images.burst
    return burst

#
#  --> Arguments <--
#

def get_sim_args(**kwargs):

    # -- init --
    args = edict()
    args.bsize = optional(kwargs,'bsize',1024) # searching batch size
    args.stride = optional(kwargs,'stride',1) # grid refinement
    args.pt = optional(kwargs,'pt',1) # patchsize, time
    args.ps = optional(kwargs,"ps",7) # patchsize, space
    args.ws = optional(kwargs,'ws',64) # spatial
    args.wf = optional(kwargs,'wf',6) # forward time
    args.wb = optional(kwargs,'wb',6) # backward time
    args.c = optional(kwargs,'c',3) # color
    args.k = optional(kwargs,'k',2) # num sim patches
    args.sigma = optional(kwargs,'sigma',0.) # noise level
    args.device = optional(kwargs,'device','cuda:0') # gpuid
    args.ref = optional(kwargs,'ref',-1) # reference frame

    return args

def get_flows(shape,**kwargs):
    flows = optional(kwargs,'flows',None)
    device = optional(kwargs,'device','cuda:0') # gpuid
    if flows is None:
        # -- init edict --
        flows = edict()

        # -- create empty values --
        t,c,h,w = shape
        fshape = (t,2,h,w)
        zflow = th.zeros(fshape,dtype=th.float32,device=device)

        # -- set values --
        flows.fflow = zflow
        flows.bflow = zflow
    else:
        flows = edict({k:th.from_numpy(v).to(device) for k,v in flows.items()})

    return flows


#
#  --> Search Locations <--
#

def get_search_inds(index,start,stop,shape,args):

    # -- bounds --
    t,c,h,w = shape
    start_b = index * args.bsize + start
    stop_b = min(( index + 1 ) * args.bsize,stop)

    # -- inds --
    srch_inds = th.arange(start_b,stop_b,args.stride,
                          dtype=th.int32,device=args.device)[:,None]
    srch_inds = get_3d_inds(srch_inds,h,w)
    srch_inds = srch_inds.contiguous()

    return srch_inds

