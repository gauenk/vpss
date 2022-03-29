"""
A file for just the sim_search kernel
"""

# -- linalg --
import numpy as np

# -- numba --
from numba import jit,njit,prange,cuda

@cuda.jit(debug=False,max_registers=64)
def sim_search_kernel(noisy,access,fflow,bflow,vals,inds,bufs,
                      tranges,n_tranges,min_tranges,ref,ps,pt,chnls,
                      bpb,nthread_loops):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = (-val-1)
        if val >= lim: val = (2*lim - val - 1)
        return val

    # -- shapes --
    nframes,color,h,w = noisy.shape
    bsize,wt,ws,ws = vals.shape
    bsize,wt,ws,ws = inds.shape
    height,width = h,w
    Z = ps*ps*pt*chnls
    nWxy = ws
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
    # top,left = h_start+hi,wstart+wi

    # -- create a range --
    wt = nWt_f + nWt_b + 1

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
        center_h,center_w = hi,wi

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        valid_t = (ti+pt-1) < nframes
        valid_t = valid_t and (ti >= 0)

        # valid_top = (top+ps-1) < height
        # valid_top = valid_top and (top >= 0)

        # valid_left = (left+ps-1) < width
        # valid_left = valid_left and (left >= 0)

        valid_anchor = valid_t# and valid_top and valid_left

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
            if tidX >= ws: continue

            for y_tile in range(nthread_loops):
                tidY = cu_tidY + blkDimY*y_tile
                if tidY >= ws: continue

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
                        cw = center_w#left
                        ch = center_h#top
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
                    if (n_ti) >= (nframes-pt+1): dist = np.inf

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

                    n_center_h = sh_start + tidX
                    n_center_w = sw_start + tidY

                    # ---------------------------
                    #      valid (search "n")
                    # ---------------------------

                    nc_row = n_center_h
                    nc_col = n_center_w
                    # nc_row = n_top + psHalf
                    # nc_col = n_left + psHalf

                    valid_t = (n_ti+pt-1) < nframes
                    valid_t = valid_t and (n_ti >= 0)
                    if ref >= 0:
                        valid_t = valid_t and (n_ti != ref)


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
                    if not(valid): dist = np.inf

                    # ---------------------------------
                    #
                    #  compute delta over patch vol.
                    #
                    # ---------------------------------

                    # -- compute difference over patch volume --
                    for pk in range(pt):
                        for pi in range(-psHalf,psHalf):
                            for pj in range(-psHalf,psHalf):

                                # -- inside entire image --
                                # vH = top+pi#bounds(top+pi,h-1)
                                # vW = left+pj#bounds(left+pj,w-1)
                                # top,left = 0,0
                                vH = bounds(center_h+pi,h)
                                vW = bounds(center_w+pj,w)
                                vT = ti + pk

                                # nH = n_top+pi#bounds(n_top+pi,h-1)
                                # nW = n_left+pj#bounds(n_left+pj,w-1)
                                nH = bounds(n_center_h+pi,h)
                                nW = bounds(n_center_w+pj,w)
                                nT = n_ti + pk

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

                    # -- smallest if matching --
                    eq_t = ti == n_ti
                    eq_h = center_h == n_center_h
                    eq_w = center_w == n_center_w
                    eq_coord = eq_t and eq_h and eq_w
                    dist = -1 if eq_coord else dist
                    vals[bidx,tidZ,tidX,tidY] = dist


                    # -- inds --
                    ind = n_ti * height * width# * color
                    # ind += (n_top) * width
                    # ind += (n_left)
                    ind += (nc_row) * width
                    ind += (nc_col)
                    # assert ind >= 0
                    inds[bidx,tidZ,tidX,tidY] = ind if dist < np.infty else -1


@cuda.jit(device=True)
def bounds(val,lim):
    if val < 0: val = (-val-1)
    if val >= lim: val = (2*lim - val - 1)
    return val

@cuda.jit(debug=False,max_registers=64)
def sim_fill_kernel(tgt,src,inds,ips,bpb):


    # -- mangling indices --
    c,h,w = tgt.shape
    thread_index = cuda.threadIdx.x
    blocksize = cuda.blockDim.x
    block_index = cuda.blockIdx.x
    global_thread_index = block_index * blocksize + thread_index

    # -- iterate over batches --
    for bpb_index in range(bpb):

        index = bpb_index + global_thread_index*bpb
        if index >= inds.shape[0]: break

        tgt_t = inds[index,0,0]
        tgt_center_h = inds[index,0,1]
        tgt_center_w = inds[index,0,2]
        src_t = inds[index,1,0]
        src_center_h = inds[index,1,1]
        src_center_w = inds[index,1,2]

        for pi in range(-ips,ips):
            for pj in range(-ips,ips):
                src_h = bounds(src_center_h + pi,h)
                src_w = bounds(src_center_w + pj,w)

                tgt_h = bounds(tgt_center_h + pi,h)
                tgt_w = bounds(tgt_center_w + pj,w)

                for ci in range(c):
                    tgt[ci,tgt_h,tgt_w] = src[src_t,ci,src_h,src_w]
