import numpy as np
import numpy.typing as npt

from preprocess_weights import hvx_preprocess_weights
from hexagon import *
from utils import print_bin, print_non_zero

VLEN        = 128   ## in bytes
# LUT_SIZE    = 128   ## in bytes
LUT_SIZE    = 16

## Note: HMX multiplication requires mandatory 32x32 tile alignment.
M_mma = K_mma = N_mma = 32
    
def dequantize_weights(
    GemmM: int, ## tiled M -> thread handles single M tile
    GemmK: int, ## tiled K
    GemmN: int, ## tiled N
    w_bits: npt.NDArray[np.uint32], 
    wdeq_lut: npt.NDArray[np.float16],
    w_hf: npt.NDArray[np.float16],
    bits: int = 4,
    g: int = 4,
    tileP: int = 512,
    tileQ: int = 64,
    vecP: int = 128,
    vecQ: int = 4,
    vecB: int = 16, ## transpose buffer
) -> npt.NDArray[np.float16]:
    M_mma = K_mma = N_mma = 32
    P = GemmM * bits
    Q = GemmK // g
    mask_4bit = 0b1111
    shift_len = 4
    # mask_vec = (np.ones(VLEN) * mask_4bit).astype(np.uint8)
    mask_vec = hvx_vector(mask_4bit, np.uint8)
    mask_vec = hvx_vector(mask_4bit, np.uint8)
    elem_per_16 = 16 // bits
    ## Restore float weights from packed weights 
    ## Step 1: Generate Look-Up Tables (LUT) based on bit position.
    ##         Example: LUT for Bit 2 (Spreading 4 bits into a 16-bit block)
    ##             0000 : 0000 0000 0000 0000
    ##             0001 : 0000 0000 0000 0100
    ##             0010 : 0000 0000 0100 0000
    ##             ...
    ##             1110 : 0100 0100 0100 0000
    ##             1111 : 0100 0100 0100 0100
    ##
    base = np.arange(16, dtype=np.uint16)
    lut_bit_0 = (base & 0b0001) + (base & 0b0010) * (1 << 3) + (base & 0b0100) * (1 << 6) + (base & 0b1000) * (1 << 9)
    ## weight restore lut
    w_b_lut = [(lut_bit_0 << b) for b in range(bits)]

    base = np.arange(16, dtype=np.float16)
    w_dq_lut = []



    ## Step 2: Restore original quantized weights using LUT mapping and bitwise OR.
    ##         Example:
    ##             Bit 3 LUT (1011) -> 1000 0000 1000 1000
    ##             Bit 2 LUT (0011) -> 0000 0000 0100 0100
    ##             Bit 1 LUT (1110) -> 0010 0010 0010 0000
    ##             Bit 0 LUT (0101) -> 0000 0001 0000 0001
    ##             ---------------------------------------
    ##             Restored         -> 1010 0011 1110 1101 (Result of ORing all layers)
    ##
    ## w_shape: (P / TileP, Q / TileQ, TileP / VecP, TileQ / VecQ, VecQ / 2, VecP) indices, elem_size = g / 8 = 0.5 bytes
    w_b = np.zeros(GemmK * GemmM, dtype = np.uint8)
    wr_buff = np.zeros(tileP * tileQ // (16 // bits), dtype = np.uint16) ## (vecP, vecP // (16 // bits))
    for tile_q in range(0, Q, tileQ):
        w_tile_base = tile_q * tileP * g // 8 
        w_b_tile_base = tile_q * tileP
        for vec_q in range(0, tileQ, vecQ):
            ## w_hf_shape: (GemmK, GemmM)
            w_b_base = w_b_tile_base + GemmM * vec_q // vecQ ## + vec_p
            w0_0123_vhp = hvx_vector_pair(0, np.uint16) ## g * vecQ == 16 ...?
            w0_89AB_vhp = hvx_vector_pair(0, np.uint16) ## vecB // vecQ == 8
            w0_4567_vhp = hvx_vector_pair(0, np.uint16)
            w0_CDEF_vhp = hvx_vector_pair(0, np.uint16)
            # w1_0123_vhp = hvx_vector_pair(0, np.uint16)
            # w1_89AB_vhp = hvx_vector_pair(0, np.uint16)
            # w1_4567_vhp = hvx_vector_pair(0, np.uint16)
            # w1_CDEF_vhp = hvx_vector_pair(0, np.uint16)
            for vec_p in range(0, tileP, vecP):
                w_base = w_tile_base + vec_p * tileQ * g // 8
                ## vecP * vecQ : 512 * 4 (P, Q) -> 128 * 16 (M, K)
                w0_lo_vb = hvx_vector_load(w_bits, w_base + vec_q * vecP * g // 8)             ## lo (lsb) 01234567 (msb)
                w0_hi_vb = hvx_vector_load(w_bits, w_base + vec_q * vecP * g // 8 + VLEN)      ## hi (lsb) 89ABCDEF (msb)
                # w1_lo_vb = hvx_vector_load(w_bits, w_base + vec_q * vecP * g // 8 + VLEN * 2)  ## lo (lsb) 01234567 (msb)
                # w1_hi_vb = hvx_vector_load(w_bits, w_base + vec_q * vecP * g // 8 + VLEN * 3)  ## hi (lsb) 89ABCDEF (msb)


                w0_lo_bo_vb = w0_lo_vb & mask_vec     ## (lsb) 0123 (msb) a  ## 128
                w0_hi_bo_vb = w0_hi_vb & mask_vec     ## (lsb) 89AB (msb) a
                w0_lo_to_vb = w0_lo_vb >> shift_len   ## (lsb) 4567 (msb) a
                w0_hi_to_vb = w0_hi_vb >> shift_len   ## (lsb) CDEF (msb) a
                # w1_lo_bo_vb = w1_lo_vb & mask_vec     ## (lsb) 0123 (msb) a  ## 128
                # w1_hi_bo_vb = w1_hi_vb & mask_vec     ## (lsb) 89AB (msb) a
                # w1_lo_to_vb = w1_lo_vb >> shift_len   ## (lsb) 4567 (msb) a
                # w1_hi_to_vb = w1_hi_vb >> shift_len   ## (lsb) CDEF (msb) a

                ## each vector pair has vecM * vecK; (vecP / 4) * (vecQ * g) 
                w0_0123_vhp |= w_b_lut[vec_p // vecP][w0_lo_bo_vb] ## (msb) 3 2 1 0 (lsb)  ## 128 x 2, (M, K): (128, 4)
                w0_89AB_vhp |= w_b_lut[vec_p // vecP][w0_hi_bo_vb] ## (msb) B A 9 8 (lsb)
                w0_4567_vhp |= w_b_lut[vec_p // vecP][w0_lo_to_vb] ## (msb) 7 6 5 4 (lsb)
                w0_CDEF_vhp |= w_b_lut[vec_p // vecP][w0_hi_to_vb] ## (msb) F E D C (lsb)
                # w1_0123_vhp |= w_b_lut[vec_p // vecP][w1_lo_bo_vb] ## (msb) 3 2 1 0 (lsb)  ## 128 x 2, (M, K): (128, 4)
                # w1_89AB_vhp |= w_b_lut[vec_p // vecP][w1_hi_bo_vb] ## (msb) B A 9 8 (lsb)
                # w1_4567_vhp |= w_b_lut[vec_p // vecP][w1_lo_to_vb] ## (msb) 7 6 5 4 (lsb)
                # w1_CDEF_vhp |= w_b_lut[vec_p // vecP][w1_hi_to_vb] ## (msb) F E D C (lsb)
            ## (M, K): (128, 16)
            w0_0123_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_0123_vhp), Q6_V_lo_W(w0_0123_vhp), -2)
            w0_4567_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_4567_vhp), Q6_V_lo_W(w0_4567_vhp), -2)
            w0_89AB_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_89AB_vhp), Q6_V_lo_W(w0_89AB_vhp), -2)
            w0_CDEF_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_CDEF_vhp), Q6_V_lo_W(w0_CDEF_vhp), -2)
            # w1_0123_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w1_0123_vhp), Q6_V_lo_W(w1_0123_vhp), -2)
            # w1_4567_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w1_4567_vhp), Q6_V_lo_W(w1_4567_vhp), -2)
            # w1_89AB_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w1_89AB_vhp), Q6_V_lo_W(w1_89AB_vhp), -2)
            # w1_CDEF_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w1_CDEF_vhp), Q6_V_lo_W(w1_CDEF_vhp), -2)
            w01 = [
                Q6_V_lo_W(w0_0123_vhp),
                Q6_V_lo_W(w0_4567_vhp),
                Q6_V_lo_W(w0_89AB_vhp),
                Q6_V_lo_W(w0_CDEF_vhp),
                # Q6_V_lo_W(w1_0123_vhp),
                # Q6_V_lo_W(w1_4567_vhp),
                # Q6_V_lo_W(w1_89AB_vhp),
                # Q6_V_lo_W(w1_CDEF_vhp),
                Q6_V_hi_W(w0_0123_vhp),
                Q6_V_hi_W(w0_4567_vhp),
                Q6_V_hi_W(w0_89AB_vhp),
                Q6_V_hi_W(w0_CDEF_vhp),
                # Q6_V_hi_W(w1_0123_vhp),
                # Q6_V_hi_W(w1_4567_vhp),
                # Q6_V_hi_W(w1_89AB_vhp),
                # Q6_V_hi_W(w1_CDEF_vhp),
            ]
            
            ## (2x64, 4) -> (4, 2x64)
            col = len(w01) // 2 
            half0 = col // 2
            for step in range(np.log2(col).astype(np.uint8)):
                half = half0 // (2**step)
                for iter_row in range(len(w01)//col):
                    w01_tile_base = iter_row * col
                    for iter_col in range(half0//half):
                        w01_base = w01_tile_base + iter_col * 2 * half
                        for i in range(half):
                            shuff_vhp = Q6_W_vshuff_VVR(w01[w01_base+half+i], w01[w01_base+i], -2)
                            w01[w01_base+i] = Q6_V_lo_W(shuff_vhp)
                            w01[w01_base+half+i] = Q6_V_hi_W(shuff_vhp)

            ## wr_buff: (tileP, tileQ // (16 // bits)), 16 for uint16
            ## wr_tile_base
            ## w01[i]: (M, K): (16, 16)
            ## w01: (M, K): (128, 16)
            ## (tileP // bits, vecQ * g) will be stored in wr_buff at a one step, (128, 16)
            ## wr_tile_base = tileP // bits * vecQ * g // elem_per_16
            wr_tile_base = vec_q // vecQ * tileP // bits * vecQ * g // elem_per_16
            for i in range(tileP // bits * vecQ * g // elem_per_16 // (VLEN // 2)): ## 2 for uint16
                wr_base = wr_tile_base + i * VLEN // 2 ## 2: 2byts for uint16
                wr_buff[wr_base:wr_base+VLEN//2] = w01[i]

        ## Step 3: Dequantize reconstructed weights into float values using a scale/offset LUT.
        ## bitserial to bitparallel conversion & partial transpose on vecP * tileQ are done
        ## now we do
        ## 1) complete transpose (concat along q axis)
        ## 2) unpack packed data to byte
        ## 3) dequantize unpacked weight with LUT
        ## (M, K): (tileP // bits, tileQ * g) are stored in wr_buff
        ## wr_buff: (tileQ // vecQ, tileP // bits // elem_per_16, vecQ * g).
        wr_tmp = [np.zeros(64, dtype=np.float16) for i in range(VLEN // (vecQ * g))]
        for iter_q in range(tileQ * g // VLEN):
            for iter_p in range(tileP // bits // ((VLEN // 2) // (vecQ * g // elem_per_16))): 
                ## tileP // bits: M
                ## (VLEN // 2) // (vecQ * g): K per vector
                wr_buff_iter_base = iter_p * VLEN // 2 + iter_q * tileP // bits // elem_per_16 * vecQ * g * (VLEN // (vecQ * g))
                for q in range(VLEN // (vecQ * g)):
                    ## wr_buff: (tileQ // vecQ, tileP // bits // elem_per_16, vecQ * g).
                    wr_buff_base = wr_buff_iter_base + q * tileP // bits // elem_per_16 * vecQ * g
                    wr_tmp[q] = wr_buff[wr_buff_base:wr_buff_base+VLEN//2] 

                ## 1) complete transpose (concat along q axis)
                ## 8x4(16) x 16 (K, M)
                # 01
                wr_0_vh = wr_tmp[0] ## 4(16) * 16
                wr_1_vh = wr_tmp[1]
                wr_2_vh = wr_tmp[2]
                wr_3_vh = wr_tmp[3]
                wr_4_vh = wr_tmp[4]
                wr_5_vh = wr_tmp[5]
                wr_6_vh = wr_tmp[6]
                wr_7_vh = wr_tmp[7]
                # 02
                wr_01_vhp = Q6_W_vshuff_VVR(wr_1_vh, wr_0_vh, -8)
                wr_23_vhp = Q6_W_vshuff_VVR(wr_3_vh, wr_2_vh, -8)
                wr_45_vhp = Q6_W_vshuff_VVR(wr_5_vh, wr_4_vh, -8)
                wr_67_vhp = Q6_W_vshuff_VVR(wr_7_vh, wr_6_vh, -8)
                # 04
                wr_0123_0_vhp = Q6_W_vshuff_VVR(Q6_V_lo_W(wr_23_vhp), Q6_V_lo_W(wr_01_vhp), -16)
                wr_0123_1_vhp = Q6_W_vshuff_VVR(Q6_V_hi_W(wr_23_vhp), Q6_V_hi_W(wr_01_vhp), -16)
                wr_4567_0_vhp = Q6_W_vshuff_VVR(Q6_V_lo_W(wr_67_vhp), Q6_V_lo_W(wr_45_vhp), -16)
                wr_4567_1_vhp = Q6_W_vshuff_VVR(Q6_V_hi_W(wr_67_vhp), Q6_V_hi_W(wr_45_vhp), -16)
                ## 32(128) * 16
                wr_01234567_0_vhp = Q6_W_vshuff_VVR(Q6_V_lo_W(wr_4567_0_vhp), Q6_V_lo_W(wr_0123_0_vhp), -32) ## 32(128) * 2
                wr_01234567_1_vhp = Q6_W_vshuff_VVR(Q6_V_hi_W(wr_4567_0_vhp), Q6_V_hi_W(wr_0123_0_vhp), -32)
                wr_01234567_2_vhp = Q6_W_vshuff_VVR(Q6_V_lo_W(wr_4567_1_vhp), Q6_V_lo_W(wr_0123_1_vhp), -32)
                wr_01234567_3_vhp = Q6_W_vshuff_VVR(Q6_V_hi_W(wr_4567_1_vhp), Q6_V_hi_W(wr_0123_1_vhp), -32)

                wr_concat_tmp = [
                    wr_01234567_0_vhp,
                    wr_01234567_1_vhp,
                    wr_01234567_2_vhp,
                    wr_01234567_3_vhp,
                ]

                # w_hf_tile_base = 
                # iter_p * ((VLEN // 2) // (vecQ * g // elem_per_16)) * GemmK:
                #   VLEN // 2: number of elments for fp16 vector
                #   vecQ * g // elem_per_16: number of elements along K axis which is packed in uint16 
                #   ((VLEN // 2) // (vecQ * g // elem_per_16)): number of rows (M) will be filled at each iteration
                #   GemmK: number of cols (K) which is stride
                # iter_q * VLEN: 
                #   VLEN of K will be filled at each iteration
                # tile_q * g: 
                #   tile_q * g of K will be filled at each tile
                w_hf_tile_base = iter_p * ((VLEN // 2) // (vecQ * g // elem_per_16)) * GemmK + iter_q * VLEN + tile_q * g
                # lut_idx_base =
                # wdeq_lut: (M // tileM, K // group_size, tileM // 4), (4 * 16)
                # g // group_size * tileP // bits // 4: stride of q tile
                lut_idx_base = iter_p * (VLEN // (vecQ * g) // 2) + iter_q * (VLEN // (vecQ * g) // 2) * tileP // bits // ((VLEN // 2) // (vecQ * g // elem_per_16)) + tile_q * g // group_size * tileP // bits // 4
                for i in range(len(wr_concat_tmp)):
                    ## 2) unpack packed data to byte
                    wr_lo_vb = Q6_V_lo_W(wr_concat_tmp[i]).view(np.uint8)
                    wr_0246_lo_vb = wr_lo_vb & mask_vec
                    wr_1357_lo_vb = wr_lo_vb >> shift_len
                    wr_0_lo_vbp = Q6_W_vshuff_VVR(wr_1357_lo_vb, wr_0246_lo_vb, -1) ## 128 * 2
                    wr_hi_vb = Q6_V_hi_W(wr_concat_tmp[i]).view(np.uint8)
                    wr_0246_hi_vb = wr_hi_vb & mask_vec
                    wr_1357_hi_vb = wr_hi_vb >> shift_len
                    wr_0_hi_vbp = Q6_W_vshuff_VVR(wr_1357_hi_vb, wr_0246_hi_vb, -1) ## 128 * 2
                    # wr_01234567_0_lo_vb = Q6_V_lo_W(wr_01234567_0_vhp).view(np.uint8)
                    # wr_0246_0_lo_vb = wr_01234567_0_lo_vb & mask_vec
                    # wr_1357_0_lo_vb = wr_01234567_0_lo_vb >> shift_len
                    # wr_01234567_0_lo_vbp = Q6_W_vshuff_VVR(wr_1357_0_lo_vb, wr_0246_0_lo_vb, -1) ## 128 * 2
                    # wr_01234567_0_hi_vb = Q6_V_hi_W(wr_01234567_0_vhp).view(np.uint8)
                    # wr_0246_0_hi_vb = wr_01234567_0_hi_vb & mask_vec
                    # wr_1357_0_hi_vb = wr_01234567_0_hi_vb >> shift_len
                    # wr_01234567_0_hi_vbp = Q6_W_vshuff_VVR(wr_1357_0_hi_vb, wr_0246_0_hi_vb, -1)

                    ## 3) dequantize unpacked weight with LUT
                    ## wdeq_lut: (M // 4, K // group_size, 4, 16)
                    wdeq_0_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_lo_W(wr_0_lo_vbp), wdeq_lut[lut_idx_base + i], 0)
                    wdeq_1_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_hi_W(wr_0_lo_vbp), wdeq_lut[lut_idx_base + i], 1)
                    wdeq_2_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_lo_W(wr_0_hi_vbp), wdeq_lut[lut_idx_base + i], 2)
                    wdeq_3_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_hi_W(wr_0_hi_vbp), wdeq_lut[lut_idx_base + i], 3)
                    # wdeq_0_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_lo_W(wr_01234567_0_lo_vbp), wdeq_lut[0], 0)
                    # wdeq_1_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_hi_W(wr_01234567_0_lo_vbp), wdeq_lut[0], 1)
                    # wdeq_2_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_lo_W(wr_01234567_0_hi_vbp), wdeq_lut[0], 2)
                    # wdeq_3_vhp = Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_hi_W(wr_01234567_0_hi_vbp), wdeq_lut[0], 3)

                    ## store
                    w_hf_base = w_hf_tile_base + i * GemmK * 4
                    w_hf[w_hf_base:w_hf_base+64] = Q6_V_lo_W(wdeq_0_vhp)
                    w_hf[w_hf_base+64:w_hf_base+128] = Q6_V_hi_W(wdeq_0_vhp)
                    w_hf_base += GemmK
                    w_hf[w_hf_base:w_hf_base+64] = Q6_V_lo_W(wdeq_1_vhp)
                    w_hf[w_hf_base+64:w_hf_base+128] = Q6_V_hi_W(wdeq_1_vhp)
                    w_hf_base += GemmK
                    w_hf[w_hf_base:w_hf_base+64] = Q6_V_lo_W(wdeq_2_vhp)
                    w_hf[w_hf_base+64:w_hf_base+128] = Q6_V_hi_W(wdeq_2_vhp)
                    w_hf_base += GemmK
                    w_hf[w_hf_base:w_hf_base+64] = Q6_V_lo_W(wdeq_3_vhp)
                    w_hf[w_hf_base+64:w_hf_base+128] = Q6_V_hi_W(wdeq_3_vhp)


    return w_hf






    # return np.random.random(w_bits.shape).astype(np.float16)

# def prefill_weight_preprocessing(w, scales, zeros):
#     return np.random.random(w.shape).astype(np.float16)

# def dequantize_weights(w, wr_lut, deq_lut):
#     return np.random.random(w.shape).astype(np.float16)


if __name__ == "__main__":
    M = 1536
    K = 1024
    N = 1024
    bits = 4
    tile_p = 512
    tile_q = 64
    vec_p = 128
    vec_q = 4
    vec_c = vec_p // 4
    g = 4
    group_size = 128

    P = M * bits ## vec_m = vec_p // bits
    Q = K // g   ## vec_k = vec_q * g
    tile_M = tile_p // bits
    tile_K = tile_q * g
    
    w_shape = (M, K) ## M, K
    # w_org = np.random.random(w_shape).astype(np.float16)
    # w = np.zeros(w_shape, dtype=np.uint8)
    # w[0:tile_M, 0:tile_K] = np.tile(np.arange(16, dtype=np.uint8), (tile_M, tile_K // 16))
    # w[0:8, 0:32] = np.arange(32*8, dtype=np.uint8).reshape(8,32) % (2**bits)
    w_org = np.random.rand(M*K).astype(np.float16).reshape(M, K)
    scale = w_org.reshape(M, K // group_size, group_size).max(axis=-1) / (2**bits - 1)
    w = np.clip(np.round(w_org.reshape(M, K // group_size, group_size) / scale[:, :, np.newaxis]).astype(np.uint8), 0, 15).reshape(M, K)
    w_deq = (w.reshape(M, K // group_size, group_size) * scale[:, :, np.newaxis]).reshape(M, K)
    # w = np.zeros(w_shape, dtype=np.uint8)
    # w[0:128, 0] = np.arange(128)%16
    # w[0, 0:256] = np.arange(256) % (2**bits)

    wdeq_lut = (np.arange(64) % (2**bits)).reshape(1, 1, 1, 4, 16) * scale.reshape(M // tile_M, tile_M // 4, 4, K // group_size).transpose(0, 3, 1, 2)[:, :, :, :, np.newaxis]
    wdeq_lut = wdeq_lut.astype(np.float16).reshape(M // tile_M, -1)
    wdeq_lut = [[wdeq_lut[m, i:i+64] for i in range(0, len(wdeq_lut[m]), 64)] for m in range(M // tile_M)]
    
    
    wp, sp = hvx_preprocess_weights(w, scale, bits=bits, tile_p=tile_p, tile_q=tile_q, vec_p=vec_p, vec_c=vec_c, g=g)
    ## x_shape: (Q / TileQ, TileQ / VecQ, VecQ, lut_size) = (Q, lut_size), elem_size = 2 bytes
    ## w_shape: (P / TileP, Q / TileQ, TileP / VecP, TileQ / VecQ, VecQ, VecP) indices, elem_size = g / 8 = 0.5 bytes
    wp = wp.view(np.uint8).reshape((P // tile_p, Q // tile_q, tile_p // vec_p, tile_q // vec_q, vec_q // 2, vec_p))
    # print(np.where(wp))
    # print_non_zero(wp)
    # print(wp.shape)
    w_hf = np.zeros((M // tile_M, tile_M * K), dtype=np.float16)
    wp = wp.reshape([P // tile_p, -1])
    # w_hf = dequantize_weights(tile_M, tile_K, N, wp[0], sp, w_hf)
    for m in range(M // tile_M):
        base = m * tile_M
        w_hf[m] = dequantize_weights(tile_M, K, N, wp[m], wdeq_lut[m], w_hf[m], bits, g, tile_p, tile_q)

        

    if np.all(w_deq == w_hf.reshape(M, K)):
        print("Passed")
    else:
        print("falied")