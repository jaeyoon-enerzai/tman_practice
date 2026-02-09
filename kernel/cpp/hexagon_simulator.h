#ifndef HVX_SIMULATOR_H
#define HVX_SIMULATOR_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

#define UNUSED(x) (void)(x)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Vector Data Types
typedef struct
{
    uint8_t data[128];
} HVX_Vector; // 128 bytes = 1024 bits

typedef struct
{
    HVX_Vector lo, hi;
} HVX_VectorPair; // 2 vectors

// Common vector sizes
constexpr int VLEN = 128; // Vector length in bytes

// ============================================================================
// MEMORY OPERATIONS
// ============================================================================

// Vector memory load
inline HVX_Vector vmem(const void *addr)
{
    HVX_Vector v;
    memcpy(v.data, addr, VLEN);
    return v;
}

// Vector memory store
inline void vmem(void *addr, const HVX_Vector &v)
{
    memcpy(addr, v.data, VLEN);
}

// L2 cache prefetch (no-op in simulation)
inline void l2fetch(const void *addr, int width, int height, int stride, int flags)
{
    // Prefetch hint - ignored in simulation
}

// Data cache prefetch (no-op in simulation)
inline void Q6_dcfetch_A(void *addr)
{
    // Prefetch hint - ignored in simulation
}

// ============================================================================
// SPLAT OPERATIONS (Broadcast scalar to vector)
// ============================================================================

// Splat byte to all vector elements
inline HVX_Vector Q6_Vb_vsplat_R(uint8_t value)
{
    HVX_Vector result;
    memset(result.data, value, VLEN);
    return result;
}

// Splat halfword (16-bit) to all vector elements
inline HVX_Vector Q6_Vh_vsplat_R(uint16_t value)
{
    HVX_Vector result;
    uint16_t *ptr = reinterpret_cast<uint16_t *>(result.data);
    for (int i = 0; i < VLEN / 2; i++)
    {
        ptr[i] = value;
    }
    return result;
}

// Splat word (32-bit) to all vector elements
inline HVX_Vector Q6_V_vsplat_R(uint32_t value)
{
    HVX_Vector result;
    uint32_t *ptr = reinterpret_cast<uint32_t *>(result.data);
    for (int i = 0; i < VLEN / 4; i++)
    {
        ptr[i] = value;
    }
    return result;
}

// ============================================================================
// LOGICAL OPERATIONS
// ============================================================================

// Vector AND
inline HVX_Vector Q6_V_vand_VV(const HVX_Vector &a, const HVX_Vector &b)
{
    HVX_Vector result;
    for (int i = 0; i < VLEN; i++)
    {
        result.data[i] = a.data[i] & b.data[i];
    }
    return result;
}

// ============================================================================
// SHIFT OPERATIONS
// ============================================================================

// Vector arithmetic shift right (halfword elements)
inline HVX_Vector Q6_Vh_vasr_VhR(const HVX_Vector &v, int shift_amount)
{
    HVX_Vector result;
    int16_t *src = (int16_t *)v.data;
    int16_t *dst = (int16_t *)result.data;

    for (int i = 0; i < VLEN / 2; i++)
    {
        dst[i] = src[i] >> shift_amount;
    }
    return result;
}

// ============================================================================
// LOOKUP TABLE OPERATIONS
// ============================================================================

// Vector lookup table - 16 entry LUT with byte indices
inline HVX_VectorPair Q6_Wh_vlut16_VbVhR_nomatch(
    const HVX_Vector &indices,
    const HVX_Vector &lut,
    int control)
{

    HVX_VectorPair result;
    const int16_t *lut_ptr = reinterpret_cast<const int16_t *>(lut.data);
    const uint8_t *idx_ptr = indices.data;
    int16_t *result_lo = reinterpret_cast<int16_t *>(result.lo.data);
    int16_t *result_hi = reinterpret_cast<int16_t *>(result.hi.data);

    // Control determines which bytes to process
    // 0: even bytes to lo, 1: even bytes to hi
    // 2: odd bytes to lo,  3: odd bytes to hi

    int offset = (control & 2) ? 1 : 0;         // odd or even bytes
    int output_offset = (control & 1) ? 64 : 0; // lo or hi output

    for (int i = 0; i < 64; i++)
    {
        uint8_t idx = idx_ptr[i * 2 + offset] & 0x0F; // 4-bit index
        int16_t value = lut_ptr[idx];

        if (control & 1)
        {
            result_hi[i] = value;
        }
        else
        {
            result_lo[i] = value;
        }
    }

    return result;
}

// ============================================================================
// VECTOR PAIR OPERATIONS
// ============================================================================

// Get low vector from vector pair
inline HVX_Vector Q6_V_lo_W(const HVX_VectorPair &pair)
{
    return pair.lo;
}

// Get high vector from vector pair
inline HVX_Vector Q6_V_hi_W(const HVX_VectorPair &pair)
{
    return pair.hi;
}

// ============================================================================
// INTEGER ARITHMETIC (Widening Operations)
// ============================================================================

// Vector add with widening: int16 -> int32
// Performs 2x64 transpose during widening
inline HVX_VectorPair Q6_Ww_vadd_VhVh(const HVX_Vector &a, const HVX_Vector &b)
{
    HVX_VectorPair result;

    const int16_t *a_ptr = reinterpret_cast<const int16_t *>(a.data);
    const int16_t *b_ptr = reinterpret_cast<const int16_t *>(b.data);
    int32_t *lo_ptr = reinterpret_cast<int32_t *>(result.lo.data);
    int32_t *hi_ptr = reinterpret_cast<int32_t *>(result.hi.data);

    // Even elements go to lo, odd elements go to hi (transpose)
    for (int i = 0; i < 32; i++)
    {
        lo_ptr[i] = static_cast<int32_t>(a_ptr[i * 2]) +
                    static_cast<int32_t>(b_ptr[i * 2]);
        hi_ptr[i] = static_cast<int32_t>(a_ptr[i * 2 + 1]) +
                    static_cast<int32_t>(b_ptr[i * 2 + 1]);
    }

    return result;
}

// Vector add with accumulation and widening
inline HVX_VectorPair Q6_Ww_vaddacc_WwVhVh(
    const HVX_VectorPair &acc,
    const HVX_Vector &a,
    const HVX_Vector &b)
{

    HVX_VectorPair result;

    const int16_t *a_ptr = reinterpret_cast<const int16_t *>(a.data);
    const int16_t *b_ptr = reinterpret_cast<const int16_t *>(b.data);
    const int32_t *acc_lo_ptr = reinterpret_cast<const int32_t *>(acc.lo.data);
    const int32_t *acc_hi_ptr = reinterpret_cast<const int32_t *>(acc.hi.data);
    int32_t *lo_ptr = reinterpret_cast<int32_t *>(result.lo.data);
    int32_t *hi_ptr = reinterpret_cast<int32_t *>(result.hi.data);

    for (int i = 0; i < 32; i++)
    {
        lo_ptr[i] = acc_lo_ptr[i] +
                    static_cast<int32_t>(a_ptr[i * 2]) +
                    static_cast<int32_t>(b_ptr[i * 2]);
        hi_ptr[i] = acc_hi_ptr[i] +
                    static_cast<int32_t>(a_ptr[i * 2 + 1]) +
                    static_cast<int32_t>(b_ptr[i * 2 + 1]);
    }

    return result;
}

// ============================================================================
// FLOATING POINT CONVERSIONS
// ============================================================================

// Convert int32 to single precision float
inline HVX_Vector Q6_Vsf_equals_Vw(const HVX_Vector &v)
{
    HVX_Vector result;
    const int32_t *src = reinterpret_cast<const int32_t *>(v.data);
    float *dst = reinterpret_cast<float *>(result.data);

    for (int i = 0; i < 32; i++)
    {
        dst[i] = static_cast<float>(src[i]);
    }
    return result;
}

// ============================================================================
// QFLOAT32 OPERATIONS (Hexagon's special FP format)
// ============================================================================

// FP16 to QFloat32 multiply (widening with 1.0)
inline HVX_VectorPair Q6_Wqf32_vmpy_VhfVhf(
    const HVX_Vector &a,
    const HVX_Vector &b)
{

    HVX_VectorPair result;

    const uint16_t *a_ptr = reinterpret_cast<const uint16_t *>(a.data);
    const uint16_t *b_ptr = reinterpret_cast<const uint16_t *>(b.data);
    float *lo_ptr = reinterpret_cast<float *>(result.lo.data);
    float *hi_ptr = reinterpret_cast<float *>(result.hi.data);

    auto fp16_to_fp32 = [](uint16_t h) -> float
    {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h & 0x7C00) >> 10;
        uint32_t mant = (h & 0x03FF);

        if (exp == 0)
        {
            if (mant == 0)
                return 0.0f;
            // Denormal
            exp = 1;
            while (!(mant & 0x0400))
            {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
        }
        else if (exp == 0x1F)
        {
            // Inf or NaN
            return (sign ? -1.0f : 1.0f) * (mant ? NAN : INFINITY);
        }

        uint32_t result_bits = sign | ((exp + 112) << 23) | (mant << 13);
        return *reinterpret_cast<float *>(&result_bits);
    };

    for (int i = 0; i < 32; i++)
    {
        lo_ptr[i] = fp16_to_fp32(a_ptr[i * 2]) * fp16_to_fp32(b_ptr[i * 2]);
        hi_ptr[i] = fp16_to_fp32(a_ptr[i * 2 + 1]) * fp16_to_fp32(b_ptr[i * 2 + 1]);
    }

    return result;
}

// QFloat32 multiply (single float)
inline HVX_Vector Q6_Vqf32_vmpy_VsfVsf(const HVX_Vector &a, const HVX_Vector &b)
{
    HVX_Vector result;
    const float *a_ptr = reinterpret_cast<const float *>(a.data);
    const float *b_ptr = reinterpret_cast<const float *>(b.data);
    float *dst = reinterpret_cast<float *>(result.data);

    for (int i = 0; i < 32; i++)
    {
        dst[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
}

// QFloat32 multiply (qfloat32)
inline HVX_Vector Q6_Vqf32_vmpy_Vqf32Vqf32(const HVX_Vector &a, const HVX_Vector &b)
{
    HVX_Vector result;
    const float *a_ptr = reinterpret_cast<const float *>(a.data);
    const float *b_ptr = reinterpret_cast<const float *>(b.data);
    float *dst = reinterpret_cast<float *>(result.data);

    for (int i = 0; i < 32; i++)
    {
        dst[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
}

// QFloat32 add (qfloat32 + qfloat32)
inline HVX_Vector Q6_Vqf32_vadd_Vqf32Vqf32(const HVX_Vector &a, const HVX_Vector &b)
{
    HVX_Vector result;
    const float *a_ptr = reinterpret_cast<const float *>(a.data);
    const float *b_ptr = reinterpret_cast<const float *>(b.data);
    float *dst = reinterpret_cast<float *>(result.data);

    for (int i = 0; i < 32; i++)
    {
        dst[i] = a_ptr[i] + b_ptr[i];
    }
    return result;
}

// QFloat32 add (qfloat32 + single float broadcast)
inline HVX_Vector Q6_Vqf32_vadd_Vqf32Vsf(const HVX_Vector &a, const HVX_Vector &b)
{
    return Q6_Vqf32_vadd_Vqf32Vqf32(a, b);
}

// ============================================================================
// SHUFFLE AND DEAL OPERATIONS (Transpose/Permute)
// ============================================================================

// Vector deal - deinterleave elements
// control: negative values indicate element size
//   -1: byte deinterleave, -2: halfword, -4: word, etc.
inline HVX_VectorPair Q6_W_vdeal_VVR(
    const HVX_Vector &a,
    const HVX_Vector &b,
    int control)
{

    HVX_VectorPair result;

    if (control == -2)
    {
        // Halfword deinterleave: separate even/odd halfwords
        const uint16_t *a_ptr = reinterpret_cast<const uint16_t *>(a.data);
        const uint16_t *b_ptr = reinterpret_cast<const uint16_t *>(b.data);
        uint16_t *lo_ptr = reinterpret_cast<uint16_t *>(result.lo.data);
        uint16_t *hi_ptr = reinterpret_cast<uint16_t *>(result.hi.data);

        // Even indices go to lo, odd to hi
        for (int i = 0; i < 32; i++)
        {
            lo_ptr[i] = b_ptr[i * 2];
            lo_ptr[i + 32] = a_ptr[i * 2];
            hi_ptr[i] = b_ptr[i * 2 + 1];
            hi_ptr[i + 32] = a_ptr[i * 2 + 1];
        }
    }
    else if (control == -1)
    {
        // Byte deinterleave
        const uint8_t *a_ptr = a.data;
        const uint8_t *b_ptr = b.data;
        uint8_t *lo_ptr = result.lo.data;
        uint8_t *hi_ptr = result.hi.data;

        for (int i = 0; i < 64; i++)
        {
            lo_ptr[i] = b_ptr[i * 2];
            lo_ptr[i + 64] = a_ptr[i * 2];
            hi_ptr[i] = b_ptr[i * 2 + 1];
            hi_ptr[i + 64] = a_ptr[i * 2 + 1];
        }
    }

    return result;
}

// Vector shuffle - interleave elements
// control: negative values indicate element size
//   -1: byte interleave, -2: halfword, -4: word, -8: doubleword, etc.
inline HVX_VectorPair Q6_W_vshuff_VVR(
    const HVX_Vector &a,
    const HVX_Vector &b,
    int control)
{

    HVX_VectorPair result;

    if (control == -1)
    {
        // Byte interleave
        const uint8_t *a_ptr = a.data;
        const uint8_t *b_ptr = b.data;
        uint8_t *lo_ptr = result.lo.data;
        uint8_t *hi_ptr = result.hi.data;

        for (int i = 0; i < 64; i++)
        {
            lo_ptr[i * 2] = b_ptr[i];
            lo_ptr[i * 2 + 1] = a_ptr[i];
            hi_ptr[i * 2] = b_ptr[i + 64];
            hi_ptr[i * 2 + 1] = a_ptr[i + 64];
        }
    }
    else if (control == -2)
    {
        // Halfword interleave
        const uint16_t *a_ptr = reinterpret_cast<const uint16_t *>(a.data);
        const uint16_t *b_ptr = reinterpret_cast<const uint16_t *>(b.data);
        uint16_t *lo_ptr = reinterpret_cast<uint16_t *>(result.lo.data);
        uint16_t *hi_ptr = reinterpret_cast<uint16_t *>(result.hi.data);

        for (int i = 0; i < 32; i++)
        {
            lo_ptr[i * 2] = b_ptr[i];
            lo_ptr[i * 2 + 1] = a_ptr[i];
            hi_ptr[i * 2] = b_ptr[i + 32];
            hi_ptr[i * 2 + 1] = a_ptr[i + 32];
        }
    }
    else if (control == -4)
    {
        // Halfword interleave
        const uint16_t *a_ptr = reinterpret_cast<const uint32_t *>(a.data);
        const uint16_t *b_ptr = reinterpret_cast<const uint32_t *>(b.data);
        uint16_t *lo_ptr = reinterpret_cast<uint32_t *>(result.lo.data);
        uint16_t *hi_ptr = reinterpret_cast<uint32_t *>(result.hi.data);

        for (int i = 0; i < 16; i++)
        {
            lo_ptr[i * 2] = b_ptr[i];
            lo_ptr[i * 2 + 1] = a_ptr[i];
            hi_ptr[i * 2] = b_ptr[i + 16];
            hi_ptr[i * 2 + 1] = a_ptr[i + 16];
        }
    }
    else if (control == -8)
    {
        // Doubleword (64-bit) interleave
        const uint64_t *a_ptr = reinterpret_cast<const uint64_t *>(a.data);
        const uint64_t *b_ptr = reinterpret_cast<const uint64_t *>(b.data);
        uint64_t *lo_ptr = reinterpret_cast<uint64_t *>(result.lo.data);
        uint64_t *hi_ptr = reinterpret_cast<uint64_t *>(result.hi.data);

        for (int i = 0; i < 8; i++)
        {
            lo_ptr[i * 2] = b_ptr[i];
            lo_ptr[i * 2 + 1] = a_ptr[i];
            hi_ptr[i * 2] = b_ptr[i + 8];
            hi_ptr[i * 2 + 1] = a_ptr[i + 8];
        }
    }
    else if (control == -16)
    {
        // 128-bit (16-byte) interleave
        const uint64_t *a_ptr = reinterpret_cast<const uint64_t *>(a.data);
        const uint64_t *b_ptr = reinterpret_cast<const uint64_t *>(b.data);
        uint64_t *lo_ptr = reinterpret_cast<uint64_t *>(result.lo.data);
        uint64_t *hi_ptr = reinterpret_cast<uint64_t *>(result.hi.data);

        for (int i = 0; i < 4; i++)
        {
            lo_ptr[i * 2] = b_ptr[i];
            lo_ptr[i * 2 + 1] = b_ptr[i + 4];
            lo_ptr[i * 2 + 8] = a_ptr[i];
            lo_ptr[i * 2 + 1 + 8] = a_ptr[i + 4];
        }
        for (int i = 0; i < 4; i++)
        {
            hi_ptr[i * 2] = b_ptr[i + 8];
            hi_ptr[i * 2 + 1] = b_ptr[i + 12];
            hi_ptr[i * 2 + 8] = a_ptr[i + 8];
            hi_ptr[i * 2 + 1 + 8] = a_ptr[i + 12];
        }
    }
    else if (control == -32)
    {
        // 256-bit (32-byte) interleave
        memcpy(result.lo.data, b.data, 32);
        memcpy(result.lo.data + 32, a.data, 32);
        memcpy(result.hi.data, b.data + 32, 32);
        memcpy(result.hi.data + 32, a.data + 32, 32);
    }

    return result;
}

// Helper for OR operation (missing from original list)
inline HVX_Vector Q6_V_vor_VV(const HVX_Vector &a, const HVX_Vector &b)
{
    HVX_Vector result;
    for (int i = 0; i < VLEN; i++)
    {
        result.data[i] = a.data[i] | b.data[i];
    }
    return result;
}

// ============================================================================
// HELPER UTILITIES
// ============================================================================

inline uint32_t _fp32_to_bits(float f)
{
    return *reinterpret_cast<uint32_t *>(&f);
}

inline uint16_t _fp16_to_bits(__fp16 h)
{
    return *reinterpret_cast<uint16_t *>(&h);
}