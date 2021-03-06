"""

Compute the differences for each block
using the centroids

"""

# -- python --
import sys,pdb
import torch
import torch as th
import torchvision
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange,cuda

from .sim_kernel import sim_search_kernel

def divUp(a,b): return (a-1)//b+1

def compute_l2norm_cuda(noisy,fflow,bflow,access,step_s,ps,ps_t,w_s,
                        nWt_f,nWt_b,step1,offset,cs,k=1):
    # todo: remove "step_s"

    # -- create output --
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- init dists --
    # (w_s = windowSpace), (w_t = windowTime)
    bsize,three = access.shape
    w_t = min(nWt_f + nWt_b + 1,t-ps_t+1)
    # w_t = min(nWt_f + nWt_b + 1,t-ps_t+1)
    # w_t = min(nWt_f + nWt_b + 1,t)
    # print(bsize,w_t,w_s,w_s)
    tf32 = torch.float32
    ti32 = torch.int32
    dists = torch.ones(bsize,w_t,w_s,w_s,dtype=tf32,device=device)
    dists *= float("inf")
    indices = -torch.ones(bsize,w_t,w_s,w_s,dtype=ti32,device=device)
    bufs = torch.zeros(bsize,3,w_t,w_s,w_s,dtype=ti32,device=device)

    # -- run launcher --
    # dists[...] = torch.inf
    # indices[...] = -1
    # print("cuda_l2: ",noisy[0,0,0,0])
    # print("[l2norm_cuda] ps,ps_t: ",ps,ps_t)
    compute_l2norm_launcher(dists,indices,fflow,bflow,access,bufs,noisy,
                            ps,ps_t,nWt_f,nWt_b,step1,offset,cs)
    # -- reshape --
    dists = rearrange(dists,'b wT wH wW -> b (wT wH wW)')
    indices = rearrange(indices,'b wT wH wW -> b (wT wH wW)')


    return dists,indices

def compute_l2norm_cuda_fast(noisy,fflow,bflow,srch_dists,srch_locs,srch_bufs,access,
                             step_s,ps,ps_t,w_s,nWt_f,nWt_b,step1,offset,cs,k=1):
    # print("srch_dists.shape: ",srch_dists.shape)
    # print("srch_locs.shape: ",srch_locs.shape)
    # print("srch_bufs.shape: ",srch_bufs.shape)
    # print("access.shape: ",access.shape)
    compute_l2norm_launcher(srch_dists,srch_locs,fflow,bflow,access,srch_bufs,noisy,
                            ps,ps_t,nWt_f,nWt_b,step1,offset,cs)
    dists = rearrange(srch_dists,'b wT wH wW -> b (wT wH wW)')
    indices = rearrange(srch_locs,'b wT wH wW -> b (wT wH wW)')
    return dists,indices


def create_frame_range(nframes,nWt_f,nWt_b,ps_t,device):
    tranges,n_tranges,min_tranges = [],[],[]
    for t_c in range(nframes-ps_t+1):

        # -- limits --
        shift_t = min(0,t_c - nWt_b) + max(0,t_c + nWt_f - nframes + ps_t)
        t_start = max(t_c - nWt_b - shift_t,0)
        t_end = min(nframes - ps_t, t_c + nWt_f - shift_t)+1

        # -- final range --
        trange = [t_c]
        trange_s = np.arange(t_c+1,t_end)
        trange_e = np.arange(t_start,t_c)[::-1]
        for t_i in range(trange_s.shape[0]):
            trange.append(trange_s[t_i])
        for t_i in range(trange_e.shape[0]):
            trange.append(trange_e[t_i])

        # -- aug vars --
        n_tranges.append(len(trange))
        min_tranges.append(np.min(trange))

        # -- add padding --
        for pad in range(nframes-len(trange)):
            trange.append(-1)

        # -- to tensor --
        trange = torch.IntTensor(trange).to(device)
        tranges.append(trange)

    tranges = torch.stack(tranges).to(device)
    n_tranges = torch.IntTensor(n_tranges).to(device)
    min_tranges = torch.IntTensor(min_tranges).to(device)

    return tranges,n_tranges,min_tranges

def compute_l2norm_launcher(dists,indices,fflow,bflow,access,bufs,noisy,
                            ps,ps_t,nWt_f,nWt_b,step1,offset,cs):

    # -- shapes --
    nframes,c,h,w = noisy.shape
    bsize,w_t,w_s,w_s = dists.shape
    bsize,w_t,w_s,w_s = indices.shape
    tranges,n_tranges,min_tranges = create_frame_range(nframes,nWt_f,nWt_b,
                                                       ps_t,noisy.device)

    # -- numbify the torch tensors --
    dists_nba = cuda.as_cuda_array(dists)
    indices_nba = cuda.as_cuda_array(indices)
    fflow_nba = cuda.as_cuda_array(fflow)
    bflow_nba = cuda.as_cuda_array(bflow)
    access_nba = cuda.as_cuda_array(access)
    bufs_nba = cuda.as_cuda_array(bufs)
    noisy_nba = cuda.as_cuda_array(noisy)
    tranges_nba = cuda.as_cuda_array(tranges)
    n_tranges_nba = cuda.as_cuda_array(n_tranges)
    min_tranges_nba = cuda.as_cuda_array(min_tranges)
    cs_nba = cuda.external_stream(cs)

    # -- batches per block --
    batches_per_block = 10
    bpb = batches_per_block

    # -- launch params --
    w_thread = min(w_s,32)
    nthread_loops = divUp(w_s,32)
    threads = (w_thread,w_thread)
    nthread_loops = w_s*w_s
    blocks = divUp(bsize,batches_per_block)
    # print(dir(cuda))
    # threads = (1,1,0)
    # blocks = (1,1,0)
    # threads,blocks = 1,1
    # print(access)
    # print("bufs.shape: ",bufs.shape)
    # print(noisy.shape)
    # print(dists.shape)
    # print(blocks,threads)
    # print(tranges.shape)
    # print(min_tranges.shape)

    try:
        # -- launch kernel --
        # print("pre.")
        # compute_l2norm_kernel[blocks,threads](dists,indices,fflow,bflow,
        #                                       access,bufs,noisy,tranges,
        #                                       n_tranges,min_tranges,
        #                                       bpb,ps,ps_t,nWt_f,nWt_b,
        #                                       nthread_loops,step1,offset)
        compute_l2norm_kernel[blocks,threads,cs_nba](dists_nba,indices_nba,
                                                     fflow_nba,bflow_nba,
                                                     access_nba,bufs_nba,
                                                     noisy_nba,tranges_nba,
                                                     n_tranges_nba,min_tranges_nba,
                                                     bpb,ps,ps_t,nWt_f,nWt_b,
                                                     nthread_loops,step1,offset)
        # torch.cuda.synchronize()
    except Exception as e:
        # pdb.set_trace()
        print("\n\n\nsome debuggin values.\n\n\n")
        for i in range(access.shape[0]):
            print(access[i])
        print(tranges)
        print(n_tranges)
        print(min_tranges)
        print("exception!")
        print(e)


def check_valid_access(access,t,h,w,ps,pt):
    """
    This code determines if the indices we accesse were valid.

    There was an error with this in an earlier version of this code.

    I am writing the function call for future users (me)
    to be aware of this access error.
    """
    pass


@cuda.jit(debug=False,max_registers=64)
def compute_l2norm_kernel(dists,inds,fflow,bflow,access,bufs,noisy,tranges,
                          n_tranges,min_tranges,bpb,ps,ps_t,nWt_f,nWt_b,
                          nthread_loops,step1,offset):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = (-val-1)
        if val >= lim: val = (2*lim - val - 1)
        return val

    def valid_frame_bounds(ti,nframes):
        leq = ti < nframes
        geq = ti >= 0
        return (leq and geq)

    def valid_top_left(n_top,n_left,h,w,ps):
        valid_top = (n_top + ps) < h
        valid_top = valid_top and (n_top >= 0)

        valid_left = (n_left + ps) < w
        valid_left = valid_left and (n_left >= 0)

        valid = valid_top and valid_left

        return valid

    # -- shapes --
    nframes,color,h,w = noisy.shape
    bsize,w_t,w_s,w_s = dists.shape
    bsize,w_t,w_s,w_s = inds.shape
    chnls = 1 if step1 else color
    height,width = h,w
    Z = ps*ps*ps_t*chnls
    nWxy = w_s

    # -- cuda threads --
    cu_tidX = cuda.threadIdx.x
    cu_tidY = cuda.threadIdx.y
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y
    tidX = cuda.threadIdx.x
    tidY = cuda.threadIdx.y

    # # -- pixel we are sim-searching for --
    # top,left = h_start+hi,w_start+wi

    # -- create a range --
    w_t = nWt_f + nWt_b + 1

    # ---------------------------
    #
    #      search frames
    #
    # ---------------------------

    # -- access with blocks and threads --
    block_start = cuda.blockIdx.x*bpb

    # -- we want enough work per thread, so we process multiple per block --
    for _bidx in range(bpb):

        # ---------------------------
        #    extract anchor pixel
        # ---------------------------

        bidx = block_start + _bidx
        if bidx >= access.shape[0]: continue
        ti = access[bidx,0]
        hi = access[bidx,1]
        wi = access[bidx,2]
        # ti,hi,wi = 0,0,0
        top,left = hi,wi

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        valid_t = (ti+ps_t-1) < nframes
        valid_t = valid_t and (ti >= 0)

        valid_top = (top+ps-1) < height
        valid_top = valid_top and (top >= 0)

        valid_left = (left+ps-1) < width
        valid_left = valid_left and (left >= 0)

        valid_anchor = valid_t and valid_top and valid_left

        # if not(valid_anchor): continue

        # ---------------------------------------
        #     searching loop for (ti,top,left)
        # ---------------------------------------

        trange = tranges[ti]
        n_trange = n_tranges[ti]
        min_trange = min_tranges[ti]

        # -- we loop over search space if needed --
        for x_tile in range(nthread_loops):
            tidX = cu_tidX + blkDimX*x_tile
            if tidX >= w_s: continue

            for y_tile in range(nthread_loops):
                tidY = cu_tidY + blkDimY*y_tile
                if tidY >= w_s: continue

                for tidZ in range(n_trange):

                    # -------------------
                    #    search frame
                    # -------------------
                    n_ti = trange[tidZ]
                    dt = trange[tidZ] - min_trange

                    # ------------------------
                    #      init direction
                    # ------------------------

                    direction = max(-1,min(1,n_ti - ti))
                    if direction != 0:
                        dtd = dt-direction
                        # if dtd >= bufs.shape[2]: continue
                        cw0 = bufs[bidx,0,dt-direction,tidX,tidY]
                        ch0 = bufs[bidx,1,dt-direction,tidX,tidY]
                        ct0 = bufs[bidx,2,dt-direction,tidX,tidY]

                        flow = fflow if direction > 0 else bflow

                        cw_f = cw0 + flow[ct0,0,ch0,cw0]
                        ch_f = ch0 + flow[ct0,1,ch0,cw0]

                        cw = max(0,min(w-1,round(cw_f)))
                        ch = max(0,min(h-1,round(ch_f)))
                        ct = n_ti
                    else:
                        cw = left
                        ch = top
                        ct = ti

                    # ----------------
                    #     update
                    # ----------------
                    # if dt >= bufs.shape[2]: continue
                    bufs[bidx,0,dt,tidX,tidY] = cw#cw_vals[ti-direction]
                    bufs[bidx,1,dt,tidX,tidY] = ch#ch_vals[t_idx-direction]
                    bufs[bidx,2,dt,tidX,tidY] = ct#ct_vals[t_idx-direction]

                    # --------------------
                    #      init dists
                    # --------------------
                    dist = 0

                    # --------------------------------
                    #   search patch's top,left
                    # --------------------------------

                    # -- target pixel we are searching --
                    if (n_ti) < 0: dist = np.inf
                    if (n_ti) >= (nframes-ps_t+1): dist = np.inf

                    # -----------------
                    #    spatial dir
                    # -----------------

                    # ch,cw = top,left
                    shift_w = min(0,cw - (nWxy-1)//2) \
                        + max(0,cw + (nWxy-1)//2 - w  + ps)
                    shift_h = min(0,ch - (nWxy-1)//2) \
                        + max(0,ch + (nWxy-1)//2 - h  + ps)

                    # -- spatial endpoints --
                    sh_start = max(0,ch - (nWxy-1)//2 - shift_h)
                    sh_end = min(h-ps,ch + (nWxy-1)//2 - shift_h)+1

                    sw_start = max(0,cw - (nWxy-1)//2 - shift_w)
                    sw_end = min(w-ps,cw + (nWxy-1)//2 - shift_w)+1

                    n_top = sh_start + tidX
                    n_left = sw_start + tidY

                    # ---------------------------
                    #      valid (search "n")
                    # ---------------------------

                    valid_t = (n_ti+ps_t-1) < nframes
                    valid_t = valid_t and (n_ti >= 0)

                    valid_top = n_top < sh_end
                    valid_top = valid_top and (n_top >= 0)

                    valid_left = n_left < sw_end
                    valid_left = valid_left and (n_left >= 0)

                    valid = valid_t and valid_top and valid_left
                    valid = valid and valid_anchor
                    if not(valid): dist = np.inf

                    # ---------------------------------
                    #
                    #  compute delta over patch vol.
                    #
                    # ---------------------------------

                    # -- compute difference over patch volume --
                    for pt in range(ps_t):
                        for pi in range(ps):
                            for pj in range(ps):

                                # -- inside entire image --
                                vH = top+pi#bounds(top+pi,h-1)
                                vW = left+pj#bounds(left+pj,w-1)
                                # vH = bounds(top+pi,h)
                                # vW = bounds(left+pj,w)
                                vT = ti + pt

                                nH = n_top+pi#bounds(n_top+pi,h-1)
                                nW = n_left+pj#bounds(n_left+pj,w-1)
                                # nH = bounds(n_top+pi,h)
                                # nW = bounds(n_left+pj,w)
                                nT = n_ti + pt

                                # -- all channels --
                                for ci in range(chnls):

                                    # -- get data --
                                    v_pix = noisy[vT][ci][vH][vW]/255.
                                    n_pix = noisy[nT][ci][nH][nW]/255.

                                    # -- compute dist --
                                    if dist < np.infty:
                                        dist += (v_pix - n_pix)**2

                    # -- dists --
                    is_zero = dist < 1e-8
                    # dist = dist/Z-offset if dist < np.infty else dist
                    # dist = dist if dist > 0 else 0.
                    # dist = 0. if is_zero else dist
                    dist /= Z
                    dist = dist-offset
                    dist = dist if dist > 0 else 0
                    # dist = abs(dist-offset)
                    dists[bidx,tidZ,tidX,tidY] = dist

                    # -- inds --
                    ind = n_ti * height * width * color
                    ind += n_top * width
                    ind += n_left
                    inds[bidx,tidZ,tidX,tidY] = ind if dist < np.infty else -1

                    # if ti == 0 and hi ==0 and wi == 0 and bidx ==0:
                    #     print("ind: ",ind)
                    #     print("n_ti: ",n_ti)
                    #     print("height: ",height)
                    #     print("width: ",width)

                    # -- final check [put self@index 0] --
                    eq_ti = n_ti == ti
                    eq_hi = n_top == top # hi
                    eq_wi = n_left == left # wi
                    eq_dim = eq_ti and eq_hi and eq_wi
                    dist = dists[bidx,tidZ,tidX,tidY]
                    dists[bidx,tidZ,tidX,tidY] = -100 if eq_dim else dist

                    # -- access pattern --
                    # access[0,dt,tidX,tidY,ti,hi,wi] = n_ti
                    # access[1,dt,tidX,tidY,ti,hi,wi] = n_top
                    # access[2,dt,tidX,tidY,ti,hi,wi] = n_left

# ------------------------------------------------------
#
#  Created to search similar patches in the center-index,
#  non-reference-frame patches.
#
# -------------------------------------------------------


def sim_cuda(noisy,access,fflow,bflow,ref,ps,pt,chnls,ws,wf,wb,k):

    # -- unpacking --
    tf32 = torch.float32
    ti32 = torch.int32
    device = noisy.device
    bsize,three = access.shape
    t,c,h,w = noisy.shape
    wt = min(wf + wb + 1,t-pt+1)

    # -- allocs --
    vals = float("inf") * th.ones(bsize,wt,ws,ws,dtype=tf32,device=device)
    inds = -th.ones(bsize,wt,ws,ws,dtype=ti32,device=device)
    bufs = th.zeros(bsize,3,wt,ws,ws,dtype=ti32,device=device)

    # -- searching helpers --
    tranges,n_tranges,min_tranges = create_frame_range(t,wf,wb,pt,device)

    # -- launching --
    sim_search_launcher(noisy,access,fflow,bflow,vals,inds,bufs,
                        tranges,n_tranges,min_tranges,ref,ps,pt,chnls)
    # -- reshape --
    vals = rearrange(vals,'b wt wh ww -> b (wt wh ww)')
    inds = rearrange(inds,'b wt wh ww -> b (wt wh ww)')

    return vals,inds


def sim_search_launcher(noisy,access,fflow,bflow,vals,inds,bufs,
                        tranges,n_tranges,min_tranges,ref,ps,pt,chnls):

    # -- numba-fy tensors --
    noisy_nba = cuda.as_cuda_array(noisy)
    access_nba = cuda.as_cuda_array(access)
    fflow_nba = cuda.as_cuda_array(fflow)
    bflow_nba = cuda.as_cuda_array(bflow)
    vals_nba = cuda.as_cuda_array(vals)
    inds_nba = cuda.as_cuda_array(inds)
    bufs_nba = cuda.as_cuda_array(bufs)
    tranges_nba = cuda.as_cuda_array(tranges)
    n_tranges_nba = cuda.as_cuda_array(n_tranges)
    min_tranges_nba = cuda.as_cuda_array(min_tranges)
    cs = th.cuda.default_stream().cuda_stream
    cs_nba = cuda.external_stream(cs)

    # -- launch params --
    batches_per_block = 10
    bpb = batches_per_block
    bsize,wt,ws,ws = inds.shape
    w_thread = min(ws,32)
    nthread_loops = divUp(ws,32)
    threads = (w_thread,w_thread)
    nthread_loops = ws*ws
    blocks = divUp(bsize,batches_per_block)

    # -- sim patch search --
    sim_search_kernel[blocks,threads,cs_nba](noisy_nba,access_nba,fflow_nba,bflow_nba,
                                             vals_nba,inds_nba,bufs_nba,tranges_nba,
                                             n_tranges_nba,min_tranges_nba,
                                             ref,ps,pt,chnls,bpb,nthread_loops)


# ------------------------------------------------------
#
#  A kernel to executing a test agains the faiss kernel
#
# -------------------------------------------------------


def python_faiss_cuda(noisy,fflow,bflow,access,step_s,ps,ps_t,w_s,
                      nWt_f,nWt_b,step1,offset,cs,k=1):

    # -- create output --
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- init dists --
    # (w_s = windowSpace), (w_t = windowTime)
    bsize,three = access.shape
    w_t = min(nWt_f + nWt_b + 1,t-ps_t+1)
    # w_t = min(nWt_f + nWt_b + 1,t-ps_t+1)
    # w_t = min(nWt_f + nWt_b + 1,t)
    # print(bsize,w_t,w_s,w_s)
    tf32 = torch.float32
    ti32 = torch.int32
    dists = torch.ones(bsize,w_t,w_s,w_s,dtype=tf32,device=device)
    dists *= float("inf")
    indices = -torch.ones(bsize,w_t,w_s,w_s,dtype=ti32,device=device)
    bufs = torch.zeros(bsize,3,w_t,w_s,w_s,dtype=ti32,device=device)

    # -- run launcher --
    python_faiss_launcher(dists,indices,fflow,bflow,access,bufs,noisy,
                          ps,ps_t,nWt_f,nWt_b,step1,offset,cs)
    # -- reshape --
    dists = rearrange(dists,'b wT wH wW -> b (wT wH wW)')
    indices = rearrange(indices,'b wT wH wW -> b (wT wH wW)')


    return dists,indices

def python_faiss_launcher(dists,indices,fflow,bflow,access,bufs,noisy,
                          ps,ps_t,nWt_f,nWt_b,step1,offset,cs):

    # -- shapes --
    nframes,c,h,w = noisy.shape
    bsize,w_t,w_s,w_s = dists.shape
    bsize,w_t,w_s,w_s = indices.shape
    tranges,n_tranges,min_tranges = create_frame_range(nframes,nWt_f,nWt_b,
                                                       ps_t,noisy.device)

    # -- numbify the torch tensors --
    dists_nba = cuda.as_cuda_array(dists)
    indices_nba = cuda.as_cuda_array(indices)
    fflow_nba = cuda.as_cuda_array(fflow)
    bflow_nba = cuda.as_cuda_array(bflow)
    access_nba = cuda.as_cuda_array(access)
    bufs_nba = cuda.as_cuda_array(bufs)
    noisy_nba = cuda.as_cuda_array(noisy)
    tranges_nba = cuda.as_cuda_array(tranges)
    n_tranges_nba = cuda.as_cuda_array(n_tranges)
    min_tranges_nba = cuda.as_cuda_array(min_tranges)
    cs_nba = cuda.external_stream(cs)
    # print(n_tranges)
    # print(min_tranges)

    # print("-- info --")
    # print(tranges[:,:14])
    # print(n_tranges)
    # print(min_tranges)

    # -- batches per block --
    batches_per_block = 10
    bpb = batches_per_block

    # -- launch params --
    w_thread = min(w_s,32)
    nthread_loops = divUp(w_s,32)
    threads = (w_thread,w_thread)
    nthread_loops = w_s*w_s
    blocks = divUp(bsize,batches_per_block)

    python_faiss_kernel[blocks,threads,cs_nba](dists_nba,indices_nba,
                                               fflow_nba,bflow_nba,
                                               access_nba,bufs_nba,
                                               noisy_nba,tranges_nba,
                                               n_tranges_nba,min_tranges_nba,
                                               bpb,ps,ps_t,nWt_f,nWt_b,
                                               nthread_loops,step1,offset)

@cuda.jit(debug=False,max_registers=64)
def python_faiss_kernel(dists,inds,fflow,bflow,access,bufs,noisy,tranges,
                        n_tranges,min_tranges,bpb,ps,ps_t,nWt_f,nWt_b,
                        nthread_loops,step1,offset):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = (-val-1)
        if val >= lim: val = (2*lim - val - 1)
        return val

    def valid_frame_bounds(ti,nframes):
        leq = ti < nframes
        geq = ti >= 0
        return (leq and geq)

    def valid_top_left(n_top,n_left,h,w,ps):
        valid_top = (n_top + ps) < h
        valid_top = valid_top and (n_top >= 0)

        valid_left = (n_left + ps) < w
        valid_left = valid_left and (n_left >= 0)

        valid = valid_top and valid_left

        return valid

    # -- shapes --
    nframes,color,h,w = noisy.shape
    bsize,w_t,w_s,w_s = dists.shape
    bsize,w_t,w_s,w_s = inds.shape
    chnls = 1 if step1 else color
    height,width = h,w
    Z = ps*ps*ps_t*chnls
    nWxy = w_s
    psHalf = ps//2
    wsHalf = (nWxy-1)//2

    # -- cuda threads --
    cu_tidX = cuda.threadIdx.x
    cu_tidY = cuda.threadIdx.y
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y
    tidX = cuda.threadIdx.x
    tidY = cuda.threadIdx.y

    # # -- pixel we are sim-searching for --
    # top,left = h_start+hi,w_start+wi

    # -- create a range --
    w_t = nWt_f + nWt_b + 1

    # ---------------------------
    #
    #      search frames
    #
    # ---------------------------

    # -- access with blocks and threads --
    block_start = cuda.blockIdx.x*bpb

    # -- we want enough work per thread, so we process multiple per block --
    for _bidx in range(bpb):

        # ---------------------------
        #    extract anchor pixel
        # ---------------------------

        bidx = block_start + _bidx
        if bidx >= access.shape[0]: continue
        ti = access[bidx,0]
        hi = access[bidx,1]-psHalf
        wi = access[bidx,2]-psHalf
        # ti,hi,wi = 0,0,0
        top,left = hi,wi

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        valid_t = (ti+ps_t-1) < nframes
        valid_t = valid_t and (ti >= 0)

        valid_top = (top+ps-1) < height
        valid_top = valid_top and (top >= 0)

        valid_left = (left+ps-1) < width
        valid_left = valid_left and (left >= 0)

        valid_anchor = valid_t and valid_top and valid_left

        # if not(valid_anchor): continue

        # ---------------------------------------
        #     searching loop for (ti,top,left)
        # ---------------------------------------

        trange = tranges[ti]
        n_trange = n_tranges[ti]
        min_trange = min_tranges[ti]

        # -- we loop over search space if needed --
        for x_tile in range(nthread_loops):
            tidX = cu_tidX + blkDimX*x_tile
            if tidX >= w_s: continue

            for y_tile in range(nthread_loops):
                tidY = cu_tidY + blkDimY*y_tile
                if tidY >= w_s: continue

                for tidZ in range(n_trange):

                    # -------------------
                    #    search frame
                    # -------------------
                    n_ti = trange[tidZ]
                    dt = trange[tidZ] - min_trange

                    # ------------------------
                    #      init direction
                    # ------------------------

                    direction = max(-1,min(1,n_ti - ti))
                    if direction != 0:
                        dtd = dt-direction
                        # if dtd >= bufs.shape[2]: continue
                        # if bidx == 10 and tidX == 0 and tidY == 0:
                        #     print("asdf")
                        #     print(top,left,ti,n_ti)
                        #     print(dt,direction,dt-direction)
                        #     print(bufs[bidx,0,0,tidX,tidY])
                        #     print(bufs[bidx,0,1,tidX,tidY])
                        #     print(bufs[bidx,0,2,tidX,tidY])
                        #     print(bufs[bidx,0,3,tidX,tidY])
                        #     print(bufs[bidx,0,4,tidX,tidY])
                        #     print(bufs[bidx,0,5,tidX,tidY])
                        #     print(bufs[bidx,0,6,tidX,tidY])
                        #     print(bufs[bidx,1,:,tidX,tidY])
                        #     print(bufs[bidx,2,:,tidX,tidY])
                        cw0 = bufs[bidx,0,dt-direction,tidX,tidY]
                        ch0 = bufs[bidx,1,dt-direction,tidX,tidY]
                        ct0 = bufs[bidx,2,dt-direction,tidX,tidY]

                        flow = fflow if direction > 0 else bflow

                        cw_f = cw0 + flow[ct0,0,ch0,cw0]
                        ch_f = ch0 + flow[ct0,1,ch0,cw0]

                        cw = round(cw_f)
                        ch = round(ch_f)
                        # cw = max(0,min(w-1,round(cw_f)))
                        # ch = max(0,min(h-1,round(ch_f)))
                        ct = n_ti
                    else:
                        cw = left
                        ch = top
                        ct = ti

                    # ----------------
                    #     update
                    # ----------------
                    # if dt >= bufs.shape[2]: continue
                    bufs[bidx,0,dt,tidX,tidY] = cw#cw_vals[ti-direction]
                    bufs[bidx,1,dt,tidX,tidY] = ch#ch_vals[t_idx-direction]
                    bufs[bidx,2,dt,tidX,tidY] = ct#ct_vals[t_idx-direction]

                    # --------------------
                    #      init dists
                    # --------------------
                    dist = 0

                    # --------------------------------
                    #   search patch's top,left
                    # --------------------------------

                    # -- target pixel we are searching --
                    if (n_ti) < 0: dist = np.inf
                    if (n_ti) >= (nframes-ps_t+1): dist = np.inf

                    # -----------------
                    #    spatial dir
                    # -----------------

                    # ch,cw = top,left
                    shift_w = min(0,cw - (nWxy-1)//2) \
                        + max(0,cw + (nWxy-1)//2 - w  + ps)
                    shift_h = min(0,ch - (nWxy-1)//2) \
                        + max(0,ch + (nWxy-1)//2 - h  + ps)

                    # -- spatial endpoints --
                    sh_start = max(0,ch - (nWxy-1)//2 - shift_h)
                    sh_end = min(h-ps,ch + (nWxy-1)//2 - shift_h)+1

                    sw_start = max(0,cw - (nWxy-1)//2 - shift_w)
                    sw_end = min(w-ps,cw + (nWxy-1)//2 - shift_w)+1

                    # TODO: remove me!
                    sh_start = ch - wsHalf
                    sw_start = cw - wsHalf

                    n_top = sh_start + tidX
                    n_left = sw_start + tidY

                    # ---------------------------
                    #      valid (search "n")
                    # ---------------------------

                    nc_row = n_top + psHalf
                    nc_col = n_left + psHalf

                    valid_t = (n_ti+ps_t-1) < nframes
                    valid_t = valid_t and (n_ti >= 0)

                    valid_top = nc_row < height#sh_end
                    valid_top = valid_top and (nc_row >= 0)

                    valid_left = nc_col < width#sw_end
                    valid_left = valid_left and (nc_col >= 0)

                    # valid_top = n_top < sh_end
                    # valid_top = valid_top and (n_top >= 0)

                    # valid_left = n_left < sw_end
                    # valid_left = valid_left and (n_left >= 0)

                    valid = valid_t and valid_top and valid_left
                    # valid = valid and valid_anchor
                    # valid = valid_t # TODO: Remove me!
                    if not(valid): dist = np.inf

                    # ---------------------------------
                    #
                    #  compute delta over patch vol.
                    #
                    # ---------------------------------

                    # -- compute difference over patch volume --
                    for pt in range(ps_t):
                        for pi in range(ps):
                            for pj in range(ps):

                                # -- inside entire image --
                                # vH = top+pi#bounds(top+pi,h-1)
                                # vW = left+pj#bounds(left+pj,w-1)
                                # top,left = 0,0
                                vH = bounds(top+pi,h)
                                vW = bounds(left+pj,w)
                                vT = ti + pt

                                # nH = n_top+pi#bounds(n_top+pi,h-1)
                                # nW = n_left+pj#bounds(n_left+pj,w-1)
                                nH = bounds(n_top+pi,h)
                                nW = bounds(n_left+pj,w)
                                nT = n_ti + pt

                                # -- all channels --
                                for ci in range(chnls):

                                    # -- get data --
                                    v_pix = noisy[vT][ci][vH][vW]/255.
                                    n_pix = noisy[nT][ci][nH][nW]/255.

                                    # -- compute dist --
                                    if dist < np.infty:
                                        dist += (v_pix - n_pix)**2

                    # -- dists --
                    is_zero = dist < 1e-8
                    # dist = dist/Z-offset if dist < np.infty else dist
                    # dist = dist if dist > 0 else 0.
                    # dist = 0. if is_zero else dist
                    dist = dist/Z if dist < np.infty else dist
                    dist = dist#-offset
                    # dist = dist if dist > 0 else 0
                    # dist = abs(dist-offset)
                    dists[bidx,tidZ,tidX,tidY] = dist

                    # -- inds --
                    ind = n_ti * height * width# * color
                    # ind += (n_top) * width
                    # ind += (n_left)
                    ind += (nc_row) * width
                    ind += (nc_col)
                    # assert ind >= 0
                    inds[bidx,tidZ,tidX,tidY] = ind if dist < np.infty else -1

                    # if ti == 0 and hi ==0 and wi == 0 and bidx ==0:
                    #     print("ind: ",ind)
                    #     print("n_ti: ",n_ti)
                    #     print("height: ",height)
                    #     print("width: ",width)

                    # -- final check [put self@index 0] --
                    # eq_ti = n_ti == ti
                    # eq_hi = n_top == top # hi
                    # eq_wi = n_left == left # wi
                    # eq_dim = eq_ti and eq_hi and eq_wi
                    # dist = dists[bidx,tidZ,tidX,tidY]
                    # dists[bidx,tidZ,tidX,tidY] = 0. if eq_dim else dist

                    # -- access pattern --
                    # access[0,dt,tidX,tidY,ti,hi,wi] = n_ti
                    # access[1,dt,tidX,tidY,ti,hi,wi] = n_top
                    # access[2,dt,tidX,tidY,ti,hi,wi] = n_left


