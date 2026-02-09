#include "hexagon_sim.h"

// For fine-grained group-wise quantization weight dequantization
template
    typename WType = uint8_t,
             typename LType = uint16_t, // LUT data type
    typename XType = __fp16,            // Output type
    int Bits = 4,
             int g = 4,
             int GroupSize = 128,
             int TileP = 512,
             int TileQ = 64,
             int VecP = 128,
             int VecQ = 4,
             int VecB = 16 >
                        inline typename std::enable_if_t
                            std::is_same<WType, uint8_t>::value &&
                                std::is_same<LType, uint16_t>::value &&
                                    std::is_same<XType, __fp16>::value,
             int >
                 hvx_dequantize_weights(
                     int32_t GemmM,
                     int32_t GemmK,
                     int32_t GemmN,
                     const WType *w_bits,
                     const XType *wdeq_lut,
                     XType *w_hf)
{

    UNUSED(GemmN);

    constexpr int32_t M_mma = 32;
    constexpr int32_t K_mma = 32;
    constexpr int32_t N_mma = 32;

    const int32_t P = GemmM * Bits;
    const int32_t Q = GemmK / g;

    constexpr int8_t mask_4bit = 0b1111;
    constexpr int8_t shift_len = 4;
    constexpr int32_t elem_per_16 = 16 / Bits;

    const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit);

    // Step 1: Generate Look-Up Tables (LUT) for bit spreading
    // LUT for each bit position to spread 4 bits into 16-bit block
    uint16_t base[16];
    for (int i = 0; i < 16; i++)
    {
        base[i] = i;
    }

    // lut_bit_0: spread bits to positions 0, 4, 8, 12
    uint16_t lut_bit_0[16];
    for (int i = 0; i < 16; i++)
    {
        lut_bit_0[i] = (base[i] & 0b0001) +
                       (base[i] & 0b0010) * (1 << 3) +
                       (base[i] & 0b0100) * (1 << 6) +
                       (base[i] & 0b1000) * (1 << 9);
    }

    // Weight restore LUTs for each bit position
    HVX_Vector w_b_lut[Bits];
    for (int b = 0; b < Bits; b++)
    {
        uint16_t *lut_ptr = reinterpret_cast<uint16_t *>(w_b_lut[b].data);
        for (int i = 0; i < 16; i++)
        {
            lut_ptr[i] = lut_bit_0[i] << b;
        }
        // Replicate pattern across entire vector
        for (int i = 16; i < VLEN / 2; i++)
        {
            lut_ptr[i] = lut_ptr[i % 16];
        }
    }

    // Restoration buffer
    uint16_t wr_buff[TileP * TileQ / elem_per_16];

    // Step 2: Restore quantized weights using LUT mapping
    for (int32_t tile_q = 0; tile_q < Q; tile_q += TileQ)
    {
        const int32_t w_tile_base = tile_q * TileP * g / 8;
        const int32_t w_b_tile_base = tile_q * TileP;

        memset(wr_buff, 0, sizeof(wr_buff));

        for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
        {
            const int32_t w_b_base = w_b_tile_base + GemmM * vec_q / VecQ;

            // Initialize accumulation vector pairs
            HVX_VectorPair w0_0123_vhp, w0_4567_vhp, w0_89AB_vhp, w0_CDEF_vhp;
            memset(&w0_0123_vhp, 0, sizeof(HVX_VectorPair));
            memset(&w0_4567_vhp, 0, sizeof(HVX_VectorPair));
            memset(&w0_89AB_vhp, 0, sizeof(HVX_VectorPair));
            memset(&w0_CDEF_vhp, 0, sizeof(HVX_VectorPair));

#pragma unroll
            for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP)
            {
                const int32_t w_base = w_tile_base + vec_p * TileQ * g / 8;

                // Load packed weights
                HVX_Vector w0_lo_vb = vmem(w_bits + w_base + vec_q * VecP * g / 8);
                HVX_Vector w0_hi_vb = vmem(w_bits + w_base + vec_q * VecP * g / 8 + VLEN);

                // Extract nibbles (4-bit values)
                HVX_Vector w0_lo_bo_vb = Q6_V_vand_VV(w0_lo_vb, mask_vec);    // bits 0-3
                HVX_Vector w0_hi_bo_vb = Q6_V_vand_VV(w0_hi_vb, mask_vec);    // bits 8-11
                HVX_Vector w0_lo_to_vb = Q6_Vh_vasr_VhR(w0_lo_vb, shift_len); // bits 4-7
                HVX_Vector w0_hi_to_vb = Q6_Vh_vasr_VhR(w0_hi_vb, shift_len); // bits 12-15

                // Apply bit-spreading LUTs and accumulate with OR
                const int lut_idx = vec_p / VecP;
                HVX_VectorPair lut_0123 = Q6_Wh_vlut16_VbVhR_nomatch(
                    w0_lo_bo_vb, w_b_lut[lut_idx], 0);
                HVX_VectorPair lut_89AB = Q6_Wh_vlut16_VbVhR_nomatch(
                    w0_hi_bo_vb, w_b_lut[lut_idx], 0);
                HVX_VectorPair lut_4567 = Q6_Wh_vlut16_VbVhR_nomatch(
                    w0_lo_to_vb, w_b_lut[lut_idx], 0);
                HVX_VectorPair lut_CDEF = Q6_Wh_vlut16_VbVhR_nomatch(
                    w0_hi_to_vb, w_b_lut[lut_idx], 0);

                // OR accumulation (bitwise reconstruction)
                w0_0123_vhp.lo = Q6_V_vor_VV(w0_0123_vhp.lo, lut_0123.lo);
                w0_0123_vhp.hi = Q6_V_vor_VV(w0_0123_vhp.hi, lut_0123.hi);
                w0_89AB_vhp.lo = Q6_V_vor_VV(w0_89AB_vhp.lo, lut_89AB.lo);
                w0_89AB_vhp.hi = Q6_V_vor_VV(w0_89AB_vhp.hi, lut_89AB.hi);
                w0_4567_vhp.lo = Q6_V_vor_VV(w0_4567_vhp.lo, lut_4567.lo);
                w0_4567_vhp.hi = Q6_V_vor_VV(w0_4567_vhp.hi, lut_4567.hi);
                w0_CDEF_vhp.lo = Q6_V_vor_VV(w0_CDEF_vhp.lo, lut_CDEF.lo);
                w0_CDEF_vhp.hi = Q6_V_vor_VV(w0_CDEF_vhp.hi, lut_CDEF.hi);
            }

            // Deal operation for transpose
            w0_0123_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_0123_vhp), Q6_V_lo_W(w0_0123_vhp), -2);
            w0_4567_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_4567_vhp), Q6_V_lo_W(w0_4567_vhp), -2);
            w0_89AB_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_89AB_vhp), Q6_V_lo_W(w0_89AB_vhp), -2);
            w0_CDEF_vhp = Q6_W_vdeal_VVR(Q6_V_hi_W(w0_CDEF_vhp), Q6_V_lo_W(w0_CDEF_vhp), -2);

            // Organize vectors for transpose
            HVX_Vector w01[8];
            w01[0] = Q6_V_lo_W(w0_0123_vhp);
            w01[1] = Q6_V_lo_W(w0_4567_vhp);
            w01[2] = Q6_V_lo_W(w0_89AB_vhp);
            w01[3] = Q6_V_lo_W(w0_CDEF_vhp);
            w01[4] = Q6_V_hi_W(w0_0123_vhp);
            w01[5] = Q6_V_hi_W(w0_4567_vhp);
            w01[6] = Q6_V_hi_W(w0_89AB_vhp);
            w01[7] = Q6_V_hi_W(w0_CDEF_vhp);

            // Multi-stage shuffle transpose: (2x64, 4) -> (4, 2x64)
            const int col = 8 / 2;     // 4
            const int half0 = col / 2; // 2

            for (int step = 0; step < 2; step++)
            { // log2(4) = 2
                const int half = half0 / (1 << step);
                for (int iter_row = 0; iter_row < 8 / col; iter_row++)
                {
                    const int w01_tile_base = iter_row * col;
                    for (int iter_col = 0; iter_col < half0 / half; iter_col++)
                    {
                        const int w01_base = w01_tile_base + iter_col * 2 * half;
                        for (int i = 0; i < half; i++)
                        {
                            HVX_VectorPair shuff_vhp = Q6_W_vshuff_VVR(
                                w01[w01_base + half + i],
                                w01[w01_base + i],
                                -2);
                            w01[w01_base + i] = Q6_V_lo_W(shuff_vhp);
                            w01[w01_base + half + i] = Q6_V_hi_W(shuff_vhp);
                        }
                    }
                }
            }

            // Store to restoration buffer
            const int32_t wr_tile_base = vec_q / VecQ * TileP / Bits * VecQ * g / elem_per_16;
            const int32_t num_vecs = TileP / Bits * VecQ * g / elem_per_16 / (VLEN / 2);

            for (int i = 0; i < num_vecs; i++)
            {
                const int32_t wr_base = wr_tile_base + i * VLEN / 2;
                memcpy(wr_buff + wr_base, w01[i].data, VLEN);
            }
        }

        // Step 3: Complete transpose, unpack, and dequantize
        XType wr_tmp[VLEN / (VecQ * g)][64];

        for (int32_t iter_q = 0; iter_q < TileQ * g / VLEN; iter_q++)
        {
            for (int32_t iter_p = 0; iter_p < TileP / Bits / ((VLEN / 2) / (VecQ * g / elem_per_16)); iter_p++)
            {

                const int32_t wr_buff_iter_base = iter_p * VLEN / 2 +
                                                  iter_q * TileP / Bits / elem_per_16 * VecQ * g * (VLEN / (VecQ * g));

                // Load tiles
                for (int q = 0; q < VLEN / (VecQ * g); q++)
                {
                    const int32_t wr_buff_base = wr_buff_iter_base +
                                                 q * TileP / Bits / elem_per_16 * VecQ * g;
                    memcpy(wr_tmp[q], wr_buff + wr_buff_base, 64 * sizeof(XType));
                }

                // Multi-stage transpose using shuffle
                HVX_Vector wr_0_vh, wr_1_vh, wr_2_vh, wr_3_vh;
                HVX_Vector wr_4_vh, wr_5_vh, wr_6_vh, wr_7_vh;

                memcpy(&wr_0_vh, wr_tmp[0], VLEN);
                memcpy(&wr_1_vh, wr_tmp[1], VLEN);
                memcpy(&wr_2_vh, wr_tmp[2], VLEN);
                memcpy(&wr_3_vh, wr_tmp[3], VLEN);
                memcpy(&wr_4_vh, wr_tmp[4], VLEN);
                memcpy(&wr_5_vh, wr_tmp[5], VLEN);
                memcpy(&wr_6_vh, wr_tmp[6], VLEN);
                memcpy(&wr_7_vh, wr_tmp[7], VLEN);

                // Pairwise shuffle
                HVX_VectorPair wr_01_vhp = Q6_W_vshuff_VVR(wr_1_vh, wr_0_vh, -8);
                HVX_VectorPair wr_23_vhp = Q6_W_vshuff_VVR(wr_3_vh, wr_2_vh, -8);
                HVX_VectorPair wr_45_vhp = Q6_W_vshuff_VVR(wr_5_vh, wr_4_vh, -8);
                HVX_VectorPair wr_67_vhp = Q6_W_vshuff_VVR(wr_7_vh, wr_6_vh, -8);

                // Quad shuffle
                HVX_VectorPair wr_0123_0_vhp = Q6_W_vshuff_VVR(
                    Q6_V_lo_W(wr_23_vhp), Q6_V_lo_W(wr_01_vhp), -16);
                HVX_VectorPair wr_0123_1_vhp = Q6_W_vshuff_VVR(
                    Q6_V_hi_W(wr_23_vhp), Q6_V_hi_W(wr_01_vhp), -16);
                HVX_VectorPair wr_4567_0_vhp = Q6_W_vshuff_VVR(
                    Q6_V_lo_W(wr_67_vhp), Q6_V_lo_W(wr_45_vhp), -16);
                HVX_VectorPair wr_4567_1_vhp = Q6_W_vshuff_VVR(
                    Q6_V_hi_W(wr_67_vhp), Q6_V_hi_W(wr_45_vhp), -16);

                // Octet shuffle
                HVX_VectorPair wr_01234567_0_vhp = Q6_W_vshuff_VVR(
                    Q6_V_lo_W(wr_4567_0_vhp), Q6_V_lo_W(wr_0123_0_vhp), -32);
                HVX_VectorPair wr_01234567_1_vhp = Q6_W_vshuff_VVR(
                    Q6_V_hi_W(wr_4567_0_vhp), Q6_V_hi_W(wr_0123_0_vhp), -32);
                HVX_VectorPair wr_01234567_2_vhp = Q6_W_vshuff_VVR(
                    Q6_V_lo_W(wr_4567_1_vhp), Q6_V_lo_W(wr_0123_1_vhp), -32);
                HVX_VectorPair wr_01234567_3_vhp = Q6_W_vshuff_VVR(
                    Q6_V_hi_W(wr_4567_1_vhp), Q6_V_hi_W(wr_0123_1_vhp), -32);

                HVX_VectorPair wr_concat_tmp[4] = {
                    wr_01234567_0_vhp,
                    wr_01234567_1_vhp,
                    wr_01234567_2_vhp,
                    wr_01234567_3_vhp};

                // Calculate base indices
                const int32_t w_hf_tile_base = iter_p * ((VLEN / 2) / (VecQ * g / elem_per_16)) * GemmK +
                                               iter_q * VLEN + tile_q * g;
                const int32_t lut_idx_base = iter_p * (VLEN / (VecQ * g) / 2) +
                                             iter_q * (VLEN / (VecQ * g) / 2) * TileP / Bits / ((VLEN / 2) / (VecQ * g / elem_per_16)) +
                                             tile_q * g / GroupSize * TileP / Bits / 4;

                // Unpack and dequantize
                for (int i = 0; i < 4; i++)
                {
                    // Unpack lo vector
                    HVX_Vector wr_lo_vb;
                    memcpy(&wr_lo_vb, &Q6_V_lo_W(wr_concat_tmp[i]), VLEN);

                    HVX_Vector wr_0246_lo_vb = Q6_V_vand_VV(wr_lo_vb, mask_vec);
                    HVX_Vector wr_1357_lo_vb = Q6_Vh_vasr_VhR(wr_lo_vb, shift_len);
                    HVX_VectorPair wr_0_lo_vbp = Q6_W_vshuff_VVR(wr_1357_lo_vb, wr_0246_lo_vb, -1);

                    // Unpack hi vector
                    HVX_Vector wr_hi_vb;
                    memcpy(&wr_hi_vb, &Q6_V_hi_W(wr_concat_tmp[i]), VLEN);

                    HVX_Vector wr_0246_hi_vb = Q6_V_vand_VV(wr_hi_vb, mask_vec);
                    HVX_Vector wr_1357_hi_vb = Q6_Vh_vasr_VhR(wr_hi_vb, shift_len);
                    HVX_VectorPair wr_0_hi_vbp = Q6_W_vshuff_VVR(wr_1357_hi_vb, wr_0246_hi_vb, -1);

                    // Dequantize using LUT
                    const XType *lut_ptr = wdeq_lut + (lut_idx_base + i) * 16;
                    HVX_Vector lut_vec = vmem(lut_ptr);

                    HVX_VectorPair wdeq_0_vhp = Q6_Wh_vlut16_VbVhR_nomatch(
                        Q6_V_lo_W(wr_0_lo_vbp), lut_vec, 0);
                    HVX_VectorPair wdeq_1_vhp = Q6_Wh_vlut16_VbVhR_nomatch(
                        Q6_V_hi_W(wr_0_lo_vbp), lut_vec, 1);
                    HVX_VectorPair wdeq_2_vhp = Q6_Wh_vlut16_VbVhR_nomatch(
                        Q6_V_lo_W(wr_0_hi_vbp), lut_vec, 2);
                    HVX_VectorPair wdeq_3_vhp = Q6_Wh_vlut16_VbVhR_nomatch(
                        Q6_V_hi_W(wr_0_hi_vbp), lut_vec, 3);

                    // Store results
                    int32_t w_hf_base = w_hf_tile_base + i * GemmK * 4;
                    vmem(w_hf + w_hf_base, Q6_V_lo_W(wdeq_0_vhp));
                    vmem(w_hf + w_hf_base + 64, Q6_V_hi_W(wdeq_0_vhp));

                    w_hf_base += GemmK;
                    vmem(w_hf + w_hf_base, Q6_V_lo_W(wdeq_1_vhp));
                    vmem(w_hf + w_hf_base + 64, Q6_V_hi_W(wdeq_1_vhp));

                    w_hf_base += GemmK;
                    vmem(w_hf + w_hf_base, Q6_V_lo_W(wdeq_2_vhp));
                    vmem(w_hf + w_hf_base + 64, Q6_V_hi_W(wdeq_2_vhp));

                    w_hf_base += GemmK;
                    vmem(w_hf + w_hf_base, Q6_V_lo_W(wdeq_3_vhp));
                    vmem(w_hf + w_hf_base + 64, Q6_V_hi_W(wdeq_3_vhp));
                }
            }
        }
    }

    return 0;
}
