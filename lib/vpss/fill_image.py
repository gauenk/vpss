
# -- python --
import sys,pdb
import torch as th
#import torchvision
import numpy as np
from einops import rearrange,repeat

# -- fill kernel --
from .sim_kernel import sim_fill_kernel

# -- numba --
from numba import jit,njit,prange,cuda

#
# Primary Logic of File
#

def fill_sim_image(sim,burst,inds,args):
    ps = args.ps
    stride = args.stride
    ips = 1
    fill_sim_image_launcher(sim,burst,inds,ips)

def fill_sim_image_launcher(sim,burst,inds,ips):

    # -- numba-fy tensors --
    sim_nba = cuda.as_cuda_array(sim)
    burst_nba = cuda.as_cuda_array(burst)
    inds_nba = cuda.as_cuda_array(inds)
    th_cs = th.cuda.default_stream().cuda_stream
    cs = cuda.external_stream(th_cs)

    # -- launch params --
    batches_per_block = 10
    bpb = batches_per_block
    bsize,k,three = inds.shape
    nthreads = 512
    nblocks = ((bsize-1)//(nthreads*bpb)) + 1

    # -- launch kernel --
    sim_fill_kernel[nblocks,nthreads,cs](sim,burst,inds,ips,bpb)
