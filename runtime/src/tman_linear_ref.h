#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>

// CPU reference implementation of TMANLinear (hvx_tbl GPTQ variant)
//
// Matches hvx_tbl<int16_t, __fp16, float, 256, 128, false, 4, 256, 4, true>
// output layout for verification.
//
// Uses UNPACKED weights (uint8, shape M x K) to avoid re-implementing the
// complex bit decomposition + interleaving + tiling + nibble packing in C++.
// Extracts individual bits from the unpacked weight to build LUT indices directly.
//
// Computation per weight group (cmp_blk_size = MIN(GroupSize/g, ActGroupSize/g) = 32):
//   1. Accumulate int16 LUT lookups across 32 q-groups (= 1 weight group) in int32
//   2. Convert int32 -> float, multiply by ls (activation scale)
//   3. For bit plane 1 only: add lb (bias = -0.5 * sum of activations in weight group)
//   4. Multiply by per-channel weight scale s (fp16)
//   5. Accumulate float result into output buffer
//
// Output layout (matches hvx_tbl c_vec_0/1/2/3 storage, verified by
// scale multiplication at hvx_funcs.h:515-519):
//   Per tile (128 channels), per bit plane (4), 128 float positions:
//     [pos  0-31 ]: c_vec_0 = even channels  0,  2, ...,  62
//     [pos 32-63 ]: c_vec_1 = even channels 64, 66, ..., 126
//     [pos 64-95 ]: c_vec_2 = odd  channels  1,  3, ...,  63
//     [pos 96-127]: c_vec_3 = odd  channels 65, 67, ..., 127
//
//   NOTE: hvx_funcs.h:475-478 comments have c_vec_1/c_vec_2 labels SWAPPED.

// fp16 (uint16_t) -> float conversion
inline float tman_fp16_to_f32(uint16_t h) {
    uint16_t sign = (h & 0x8000u) >> 15;
    uint16_t exp  = (h & 0x7C00u) >> 10;
    uint16_t frac = (h & 0x03FFu);

    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * std::ldexp(static_cast<float>(frac), -24);
    }
    if (exp == 31) {
        if (frac == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    float mant = 1.0f + static_cast<float>(frac) / 1024.0f;
    int e = static_cast<int>(exp) - 15;
    float val = std::ldexp(mant, e);
    return sign ? -val : val;
}

// Map channel index (0..127 within a tile) to output buffer position
// within a 128-float vec_p block.
//
// Derivation from vlut16 + widening add in hvx_tbl:
//   Weight is byte-interleaved: [ch0, ch64, ch1, ch65, ...]
//   vlut16 separates even bytes -> channels 0..63, odd bytes -> channels 64..127
//   Widening add splits even elements -> lo_W, odd elements -> hi_W
//
//   c_vec_0 = lo_W(c_vec_lo) = even-indexed of channels 0..63  = ch0,ch2,...,ch62
//   c_vec_1 = lo_W(c_vec_hi) = even-indexed of channels 64..127 = ch64,ch66,...,ch126
//   c_vec_2 = hi_W(c_vec_lo) = odd-indexed of channels 0..63   = ch1,ch3,...,ch63
//   c_vec_3 = hi_W(c_vec_hi) = odd-indexed of channels 64..127  = ch65,ch67,...,ch127
inline int32_t ch_to_pos(int32_t ch) {
    if (ch < 64)
        return (ch % 2 == 0) ? (ch / 2) : (64 + ch / 2);
    else
        return (ch % 2 == 0) ? (32 + (ch - 64) / 2) : (96 + (ch - 64) / 2);
}

// Compute the output buffer size in bytes
// = M * bits * sizeof(float)
inline int32_t tman_linear_ref_bufsize(int32_t gemm_m, int32_t bits) {
    return gemm_m * bits * (int32_t)sizeof(float);
}

// CPU reference for TMANLinear
//
// Parameters (matching hvx_preprocess_weights terminology):
//   gemm_m     : output features M (= 2048)
//   gemm_k     : input features K  (= 8192)
//   bits       : weight quantization bits (= 4)
//   group_size : weight quantization group size in K-domain (= 128)
//
// Inputs:
//   precompute_buf : raw output from precompute_ref() [l, ls, lb concatenated]
//   w_unpacked     : uint8, shape (M, K), row-major. Each element is a quantized
//                    weight value (0..2^bits-1). From w_unpacked.bin.
//   s_unpacked     : uint16 (fp16), shape (M, K/group_size), row-major.
//                    Per-channel per-weight-group scale. From s_unpacked.bin.
//
// Output:
//   output         : float buffer, size = M * bits * sizeof(float) bytes.
//                    Layout matches hvx_tbl output (interleaved channels, per-tile per-bitplane).
inline void tman_linear_ref(
    int32_t gemm_m,
    int32_t gemm_k,
    int32_t bits,
    int32_t group_size,
    const uint8_t* precompute_buf,
    const uint8_t* w_unpacked,
    const uint16_t* s_unpacked,
    float* output
)
{
    // Constants matching hvx_tbl template parameters
    constexpr int32_t g = 4;              // LUT group size (LutG)
    constexpr int32_t lut_size = 16;      // 2^g entries per LUT
    constexpr int32_t act_group_size = 256;
    constexpr int32_t vec_p = 128;        // HVX vector register size in bytes

    const int32_t M = gemm_m;
    const int32_t K = gemm_k;
    const int32_t Q = K / g;                           // total q-groups (= K / g)
    const int32_t q_wgt = group_size / g;               // q-groups per weight group (= 32)
    const int32_t q_act = act_group_size / g;           // q-groups per activation group (= 64)
    const int32_t num_wgt_groups = K / group_size;      // total weight groups (= 64)
    const int32_t channels_per_tile = vec_p;            // 128 channels per tile

    // Parse precompute buffer layout (same as precompute_ref.h)
    const int32_t l_count   = Q * lut_size;             // total int16 LUT entries
    const int32_t num_scale = K / act_group_size;       // number of activation scales
    const int32_t ls_pad    = std::max(num_scale, (int32_t)(128 / (int32_t)sizeof(float)));

    const int16_t* l_ptr  = reinterpret_cast<const int16_t*>(precompute_buf);
    const float*   ls_ptr = reinterpret_cast<const float*>(l_ptr + l_count);
    const float*   lb_ptr = ls_ptr + ls_pad;

    // Zero-initialize output
    const int32_t out_floats = M * bits;
    std::memset(output, 0, out_floats * sizeof(float));

    // Process each output channel
    for (int32_t m = 0; m < M; m++)
    {
        const int32_t tile = m / channels_per_tile;
        const int32_t ch_in_tile = m % channels_per_tile;
        const int32_t pos = ch_to_pos(ch_in_tile);

        // For each weight group (cmp_blk_size = MIN(GroupSize/g, ActGroupSize/g) = q_wgt = 32)
        for (int32_t wg = 0; wg < num_wgt_groups; wg++)
        {
            const int32_t qq_start = wg * q_wgt;       // first q-group in this weight group
            const int32_t ag = qq_start / q_act;        // activation group index

            // Accumulate LUT lookups per bit plane in int32
            int32_t acc[4] = {0, 0, 0, 0};

            for (int32_t local_q = 0; local_q < q_wgt; local_q++)
            {
                const int32_t qq = qq_start + local_q;  // absolute q-group index

                for (int32_t b = 0; b < bits; b++)
                {
                    // Build g-bit LUT index from weight bit b
                    // For each of g=4 consecutive K positions, extract bit b
                    uint8_t idx = 0;
                    for (int32_t j = 0; j < g; j++)
                    {
                        int32_t k = qq * g + j;
                        idx |= ((w_unpacked[m * K + k] >> b) & 1) << j;
                    }

                    // Look up precomputed int16 LUT value (interleaved-pair format)
                    //   pair = (q within activation group) / 2
                    //   within = (q within activation group) % 2
                    //   base = activation_group_start * lut_size + pair * lut_size * 2
                    const int32_t q_in_ag = qq % q_act;
                    const int32_t pair = q_in_ag / 2;
                    const int32_t within = q_in_ag % 2;
                    const int32_t base = (qq - q_in_ag) * lut_size + pair * lut_size * 2;

                    acc[b] += (int32_t)l_ptr[base + idx * 2 + within];
                }
            }

            // Apply scales (matching hvx_tbl block tail at cmp_blk_tail)
            const float ls_val = ls_ptr[ag];
            const float lb_val = lb_ptr[wg];
            const float s_val = tman_fp16_to_f32(s_unpacked[m * num_wgt_groups + wg]);

            for (int32_t b = 0; b < bits; b++)
            {
                float val = (float)acc[b] * ls_val;
                if (b == 1) val += lb_val;   // bias only for bit plane 1
                val *= s_val;

                // Output index: (tile * bits_per_tile + bitplane) * vec_p + interleaved_pos
                const int32_t out_idx = (tile * bits + b) * vec_p + pos;
                output[out_idx] += val;
            }
        }
    }

    std::cout << "  tman_linear_ref: M=" << M << " K=" << K
              << " bits=" << bits << " group_size=" << group_size
              << " num_wgt_groups=" << num_wgt_groups
              << " output_floats=" << out_floats << "\n";
}
