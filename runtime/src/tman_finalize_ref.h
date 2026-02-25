#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

// CPU reference implementation of TMANFinalize (hvx_bit_serial)
//
// Takes TMANLinear output (qf32 data as IEEE float from CPU ref) in
// the interleaved per-tile per-bitplane layout and produces fp16
// output in natural channel order.
//
// Algorithm (matching hvx_bit_serial):
//   result = bp0 * 0.5 + bp1 * 1.0 + bp2 * 2.0 + bp3 * 4.0
//   Then de-interleave even/odd channels (TAG1 reversal).

// float32 -> fp16 (uint16_t) conversion
inline uint16_t tman_f32_to_fp16(float val) {
    uint32_t f;
    std::memcpy(&f, &val, sizeof(f));

    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (f >> 13) & 0x3FF;

    if (exp <= 0) return (uint16_t)sign;          // flush to zero
    if (exp >= 31) return (uint16_t)(sign | 0x7C00); // infinity

    return (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
}

// Output buffer size in bytes = M * sizeof(fp16)
inline int32_t tman_finalize_ref_bufsize(int32_t gemm_m) {
    return gemm_m * (int32_t)sizeof(uint16_t);
}

// CPU reference for TMANFinalize
//
// Parameters:
//   gemm_m   : output features M (= 2048)
//   bits     : weight quantization bits (= 4)
//
// Input:
//   c_input  : float buffer from tman_linear_ref(), size = M * bits floats.
//              Layout: per tile (128 channels), per bitplane (bits), 128 floats.
//              Within each 128-float bitplane block:
//                [0-31]:   c_vec_0 = even channels 0,2,...,62
//                [32-63]:  c_vec_1 = even channels 64,66,...,126
//                [64-95]:  c_vec_2 = odd channels 1,3,...,63
//                [96-127]: c_vec_3 = odd channels 65,67,...,127
//
// Output:
//   y_output : uint16_t (fp16) buffer, size = M values.
//              Natural channel order: ch0, ch1, ch2, ..., ch2047.
inline void tman_finalize_ref(
    int32_t gemm_m,
    int32_t bits,
    const float* c_input,
    uint16_t* y_output
)
{
    constexpr int32_t channels_per_tile = 128;  // VecP in hvx_bit_serial
    const int32_t num_tiles = gemm_m / channels_per_tile;

    for (int32_t tile = 0; tile < num_tiles; tile++)
    {
        const int32_t tile_base = tile * bits * channels_per_tile;  // 512 floats per tile

        // Bit-serial weighted sum for each of 4 sub-vectors (32 elements each)
        float bitsum[4][32];
        for (int32_t sv = 0; sv < 4; sv++)
        {
            for (int32_t e = 0; e < 32; e++)
            {
                float val = c_input[tile_base + 0 * channels_per_tile + sv * 32 + e] * 0.5f;  // bp0 * 0.5
                val += c_input[tile_base + 1 * channels_per_tile + sv * 32 + e];               // bp1 * 1.0
                if (bits >= 3)
                    val += c_input[tile_base + 2 * channels_per_tile + sv * 32 + e] * 2.0f;    // bp2 * 2.0
                if (bits >= 4)
                    val += c_input[tile_base + 3 * channels_per_tile + sv * 32 + e] * 4.0f;    // bp3 * 4.0
                bitsum[sv][e] = val;
            }
        }

        // De-interleave (TAG1 reversal):
        //   Q6_Vhf_equals_Wqf32(combine(c_bits[2], c_bits[0]))
        //     lo=c_bits[0] (even ch 0-62), hi=c_bits[2] (odd ch 1-63)
        //     → interleaved: ch0, ch1, ch2, ch3, ..., ch62, ch63
        //
        //   Q6_Vhf_equals_Wqf32(combine(c_bits[3], c_bits[1]))
        //     lo=c_bits[1] (even ch 64-126), hi=c_bits[3] (odd ch 65-127)
        //     → interleaved: ch64, ch65, ..., ch126, ch127
        const int32_t out_base = tile * channels_per_tile;

        for (int32_t e = 0; e < 32; e++)
        {
            // channels 0-63: interleave bitsum[0] (even) and bitsum[2] (odd)
            y_output[out_base + e * 2]     = tman_f32_to_fp16(bitsum[0][e]);  // ch 0,2,4,...,62
            y_output[out_base + e * 2 + 1] = tman_f32_to_fp16(bitsum[2][e]);  // ch 1,3,5,...,63
        }
        for (int32_t e = 0; e < 32; e++)
        {
            // channels 64-127: interleave bitsum[1] (even) and bitsum[3] (odd)
            y_output[out_base + 64 + e * 2]     = tman_f32_to_fp16(bitsum[1][e]);  // ch 64,66,...,126
            y_output[out_base + 64 + e * 2 + 1] = tman_f32_to_fp16(bitsum[3][e]);  // ch 65,67,...,127
        }
    }

    std::cout << "  tman_finalize_ref: M=" << gemm_m << " bits=" << bits
              << " num_tiles=" << num_tiles
              << " output_fp16=" << gemm_m << "\n";
}
