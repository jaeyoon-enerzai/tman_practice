#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

// CPU reference implementation of TMANPrecompute (hvx_lut_ctor)
//
// Matches hvx_lut_ctor<int16_t, __fp16, 256, GroupSize, false, 4>
// output layout for verification.
//
// Output buffer layout (same as HVX version):
//   [l_ptr          ][ls_ptr         ][lb_ptr         ]
//   int16 LUT table   float scales     float biases
//   Q*16 entries       max(nscales,32)  Q/q_wgt values
//
// LUT memory layout (interleaved-pair, matching vlut16 format):
//   Pairs are stored sequentially: pair 0, pair 1, ..., pair 31
//   Each pair (g_{2k}, g_{2k+1}) occupies 32 int16 = 64 bytes:
//     [g_{2k}_L0, g_{2k+1}_L0, g_{2k}_L1, g_{2k+1}_L1, ..., g_{2k}_L15, g_{2k+1}_L15]
//
// LUT entry formula:
//   LUT[i] = sum_j( (bit_j(i) ? +1 : -1) * x[group*4 + j] )  for j=0..3

// fp32 -> fp16 -> fp32 roundtrip to match HTP precision
inline float fp16_round(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;

    uint32_t sign = (v.u >> 31) & 1;
    int32_t  exp  = ((v.u >> 23) & 0xFF) - 127;
    uint32_t man  = v.u & 0x7FFFFF;

    // Overflow -> fp16 max
    if (exp > 15)
        return sign ? -65504.0f : 65504.0f;
    // Underflow -> zero
    if (exp < -24)
        return 0.0f;

    // Truncate mantissa to 10 bits (drop lower 13 bits)
    man = man & 0x7FE000;

    v.u = (sign << 31) | ((uint32_t)(exp + 127) << 23) | man;
    return v.f;
}

inline void precompute_ref(
    int32_t gemm_k,
    int32_t group_size,        // weight quantization group size (e.g. 128)
    const float* x_fp32,       // input: fp32 activations, length gemm_k
    uint8_t* output,           // output: raw buffer (same layout as HVX)
    bool use_fp16_round = true // set false to skip fp16 rounding of input
)
{
    constexpr int32_t g = 4;
    constexpr int32_t lut_size = 16;
    constexpr int32_t act_group_size = 256;
    constexpr float max_int16 = 32767.0f;

    const int32_t Q         = gemm_k / g;
    const int32_t q_act     = act_group_size / g;   // 64
    const int32_t q_wgt     = group_size / g;        // 32
    const int32_t l_count   = Q * lut_size;          // total int16 LUT entries
    const int32_t num_scale = gemm_k / act_group_size;
    const int32_t ls_pad    = std::max(num_scale, (int32_t)(128 / sizeof(float)));

    // Pointers into output buffer (matching TMANPrecompute.cpp layout)
    int16_t* l  = reinterpret_cast<int16_t*>(output);
    float*   ls = reinterpret_cast<float*>(l + l_count);
    float*   lb = ls + ls_pad;

    // Prepare fp16-rounded activations
    std::vector<float> x(gemm_k);
    for (int32_t i = 0; i < gemm_k; i++)
        x[i] = use_fp16_round ? fp16_round(x_fp32[i]) : x_fp32[i];

    // Process each activation group
    for (int32_t gq = 0; gq < Q; gq += q_act)
    {
        // ---- Step 1: Compute scale (ls) ----
        // max of (|e0| + |e1| + |e2| + |e3|) across all groups in this act group
        float max_sabs = 0.0f;
        for (int32_t q = 0; q < q_act; q++)
        {
            float sabs = 0.0f;
            for (int32_t j = 0; j < g; j++)
                sabs += fabsf(x[(gq + q) * g + j]);
            max_sabs = std::max(max_sabs, sabs);
        }

        float ls_val = max_sabs / max_int16;
        ls[gq / q_act] = ls_val;
        float inv_ls = ls_val > 0.0f ? 1.0f / ls_val : 0.0f;

        // ---- Step 2: Build LUT, quantize, store in interleaved-pair layout ----
        for (int32_t q = 0; q < q_act; q++)
        {
            float e[g];
            for (int32_t j = 0; j < g; j++)
                e[j] = x[(gq + q) * g + j];

            int16_t lut_q[lut_size];
            for (int32_t i = 0; i < lut_size; i++)
            {
                // LUT[i] = sum of (+/- e_j) based on bit j of i
                float val = 0.0f;
                for (int32_t j = 0; j < g; j++)
                    val += (i & (1 << j)) ? e[j] : -e[j];

                // Quantize: scale, truncate to int32, saturate to int16
                float scaled = val * inv_ls;
                int32_t ival = (int32_t)scaled;
                ival = std::max(std::min(ival, (int32_t)32767), (int32_t)-32768);
                lut_q[i] = (int16_t)ival;
            }

            // Store position:
            //   pair = q / 2,  within = q % 2
            //   pos = gq*16 + pair*32 + entry*2 + within
            int32_t pair   = q / 2;
            int32_t within = q % 2;
            int32_t base   = gq * lut_size + pair * lut_size * 2;
            for (int32_t i = 0; i < lut_size; i++)
                l[base + i * 2 + within] = lut_q[i];
        }

        // ---- Step 3: Compute bias (lb) per weight group ----
        // lb = -0.5 * sum of all activations in the weight group
        for (int32_t w = 0; w < q_act / q_wgt; w++)
        {
            float sum = 0.0f;
            for (int32_t q = 0; q < q_wgt; q++)
                for (int32_t j = 0; j < g; j++)
                    sum += x[(gq + w * q_wgt + q) * g + j];

            lb[gq / q_wgt + w] = -sum * 0.5f;
        }
    }
}

// Compute the total output buffer size in bytes
inline int32_t precompute_ref_bufsize(int32_t gemm_k, int32_t group_size)
{
    constexpr int32_t g = 4;
    constexpr int32_t lut_size = 16;
    constexpr int32_t act_group_size = 256;

    int32_t Q         = gemm_k / g;
    int32_t l_bytes   = Q * lut_size * (int32_t)sizeof(int16_t);
    int32_t num_scale = gemm_k / act_group_size;
    int32_t ls_bytes  = std::max(num_scale, (int32_t)(128 / (int32_t)sizeof(float))) * (int32_t)sizeof(float);
    int32_t lb_count  = Q / (group_size / g);
    int32_t lb_bytes  = std::max((int32_t)(lb_count * (int32_t)sizeof(float)), (int32_t)128);

    return l_bytes + ls_bytes + lb_bytes;
}
