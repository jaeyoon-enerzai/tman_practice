import numpy as np

VLEN        = 128   ## in bytes

hvx_vector_load = lambda src, start: src[start:start+VLEN//src.dtype.itemsize]

# hvx_vector_store = lambda dest, src, start: dest[start:start+VLEN//dest.dtype.itemsize]

hvx_vector = lambda value, dtype: np.ones(VLEN//np.dtype(dtype).itemsize, dtype=dtype) * value

hvx_vector_pair = lambda value, dtype: np.ones(2*VLEN//np.dtype(dtype).itemsize, dtype=dtype) * value

Q6_V_lo_W = lambda w: w[:len(w)//2]

Q6_V_hi_W = lambda w: w[len(w)//2:]

def Q6_W_vdeal_VVR(v1, v0, r):
    dtype_org = v0.dtype
    return np.concatenate([v0, v1]).view(np.uint8).reshape(-1, 2, -r).transpose(1, 0, 2).reshape(-1).view(dtype_org)

def Q6_W_vshuff_VVR(v1, v0, r):
    dtype_org = v0.dtype
    return np.concatenate([v0, v1]).view(np.uint8).reshape(2, -1, -r).transpose(1, 0, 2).reshape(-1).view(dtype_org)

Q6_Wh_vlut16_VbVhR_nomatch = lambda vb, vh, r: vh[r*16:(r+1)*16][vb]