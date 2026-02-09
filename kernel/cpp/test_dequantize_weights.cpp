// test_hvx_dequantize.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iomanip>

// Include your HVX simulator functions here
#include "hexagon_simulator.h"

// JSON parsing for metadata (simple implementation)
struct Metadata
{
    int M, K, N;
    int bits, tile_p, tile_q, vec_p, vec_q, g, group_size;
    int tile_M, tile_K, P, Q;
    int wp_rows, wp_cols;
    int lut_rows, lut_cols;
};

Metadata parse_metadata(const std::string &filename)
{
    std::ifstream file(filename);
    Metadata meta = {};

    std::string line;
    while (std::getline(file, line))
    {
        // Simple JSON parsing (you can use a library for robustness)
        if (line.find("\"M\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"M\": %d", &meta.M);
        }
        else if (line.find("\"K\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"K\": %d", &meta.K);
        }
        else if (line.find("\"N\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"N\": %d", &meta.N);
        }
        else if (line.find("\"bits\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"bits\": %d", &meta.bits);
        }
        else if (line.find("\"tile_p\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"tile_p\": %d", &meta.tile_p);
        }
        else if (line.find("\"tile_q\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"tile_q\": %d", &meta.tile_q);
        }
        else if (line.find("\"vec_p\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"vec_p\": %d", &meta.vec_p);
        }
        else if (line.find("\"vec_q\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"vec_q\": %d", &meta.vec_q);
        }
        else if (line.find("\"g\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"g\": %d", &meta.g);
        }
        else if (line.find("\"group_size\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"group_size\": %d", &meta.group_size);
        }
        else if (line.find("\"tile_M\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"tile_M\": %d", &meta.tile_M);
        }
        else if (line.find("\"tile_K\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"tile_K\": %d", &meta.tile_K);
        }
        else if (line.find("\"P\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"P\": %d", &meta.P);
        }
        else if (line.find("\"Q\":") != std::string::npos)
        {
            sscanf(line.c_str(), " \"Q\": %d", &meta.Q);
        }
        else if (line.find("\"wp_shape\":") != std::string::npos)
        {
            // Read next line for array values
            std::getline(file, line);
            sscanf(line.c_str(), " %d,", &meta.wp_rows);
            std::getline(file, line);
            sscanf(line.c_str(), " %d", &meta.wp_cols);
        }
        else if (line.find("\"wdeq_lut_shape\":") != std::string::npos)
        {
            std::getline(file, line);
            sscanf(line.c_str(), " %d,", &meta.lut_rows);
            std::getline(file, line);
            sscanf(line.c_str(), " %d", &meta.lut_cols);
        }
    }

    return meta;
}

template <typename T>
std::vector<T> load_binary_file(const std::string &filename, size_t expected_size = 0)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(T);
    if (expected_size > 0 && num_elements != expected_size)
    {
        std::cerr << "Warning: File size mismatch. Expected " << expected_size
                  << " elements, got " << num_elements << std::endl;
    }

    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char *>(data.data()), file_size);

    std::cout << "Loaded " << filename << ": " << num_elements << " elements ("
              << file_size / 1024.0 << " KB)" << std::endl;

    return data;
}

// FP16 conversion helper (if __fp16 not available)
#ifndef __ARM_FP16_FORMAT_IEEE
typedef uint16_t __fp16;

inline float fp16_to_fp32(uint16_t h)
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
        return (sign ? -1.0f : 1.0f) * (mant ? NAN : INFINITY);
    }

    uint32_t result_bits = sign | ((exp + 112) << 23) | (mant << 13);
    return *reinterpret_cast<float *>(&result_bits);
}

inline uint16_t fp32_to_fp16(float f)
{
    uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 112;
    uint32_t mant = bits & 0x007FFFFF;

    if (exp <= 0)
    {
        if (exp < -10)
            return sign;
        mant = (mant | 0x00800000) >> (1 - exp);
        return sign | (mant >> 13);
    }
    else if (exp >= 0x1F)
    {
        return sign | 0x7C00;
    }

    return sign | (exp << 10) | (mant >> 13);
}
#endif

bool compare_fp16_arrays(const __fp16 *result, const __fp16 *reference,
                         size_t size, float rtol = 1e-3, float atol = 1e-5)
{
    size_t num_errors = 0;
    float max_error = 0.0f;
    size_t max_error_idx = 0;

    for (size_t i = 0; i < size; i++)
    {
        float res = fp16_to_fp32(reinterpret_cast<const uint16_t *>(result)[i]);
        float ref = fp16_to_fp32(reinterpret_cast<const uint16_t *>(reference)[i]);

        float diff = std::abs(res - ref);
        float threshold = atol + rtol * std::abs(ref);

        if (diff > threshold)
        {
            if (num_errors < 10)
            { // Print first 10 errors
                std::cout << "Mismatch at index " << i << ": "
                          << "result=" << res << ", reference=" << ref
                          << ", diff=" << diff << std::endl;
            }
            num_errors++;

            if (diff > max_error)
            {
                max_error = diff;
                max_error_idx = i;
            }
        }
    }

    if (num_errors > 0)
    {
        std::cout << "\nTotal mismatches: " << num_errors << " / " << size
                  << " (" << (100.0 * num_errors / size) << "%)" << std::endl;
        std::cout << "Max error: " << max_error << " at index " << max_error_idx << std::endl;
        return false;
    }

    return true;
}

int main()
{
    std::cout << "=== HVX Weight Dequantization Test ===" << std::endl;

    // Load metadata
    std::cout << "\nLoading metadata..." << std::endl;
    Metadata meta = parse_metadata("metadata.json");

    std::cout << "Configuration:" << std::endl;
    std::cout << "  M=" << meta.M << ", K=" << meta.K << ", N=" << meta.N << std::endl;
    std::cout << "  bits=" << meta.bits << ", g=" << meta.g << std::endl;
    std::cout << "  tile_M=" << meta.tile_M << ", tile_K=" << meta.tile_K << std::endl;
    std::cout << "  tile_p=" << meta.tile_p << ", tile_q=" << meta.tile_q << std::endl;

    // Load input data
    std::cout << "\nLoading input data..." << std::endl;
    auto wp = load_binary_file<uint8_t>("data/wp.bin", meta.wp_rows * meta.wp_cols);
    auto wdeq_lut = load_binary_file<__fp16>("data/wdeq_lut.bin", meta.lut_rows * meta.lut_cols);
    auto w_deq_ref = load_binary_file<__fp16>("data/w_deq_ref.bin", meta.M * meta.K);

    // Allocate output buffer
    std::vector<__fp16> w_hf(meta.M * meta.K, 0);

    // Run dequantization for each M tile
    std::cout << "\nRunning dequantization..." << std::endl;
    const int num_m_tiles = meta.M / meta.tile_M;

    for (int m = 0; m < num_m_tiles; m++)
    {
        std::cout << "Processing M tile " << m + 1 << "/" << num_m_tiles << "..." << std::endl;

        const uint8_t *wp_ptr = wp.data() + m * meta.wp_cols;
        const __fp16 *lut_ptr = wdeq_lut.data() + m * meta.lut_cols;
        __fp16 *out_ptr = w_hf.data() + m * meta.tile_M * meta.K;

        int status = hvx_dequantize_weights
            uint8_t,            // WType
            uint16_t,           // LType
            __fp16,             // XType
            4,                  // Bits
            4,                  // g
            128,                // GroupSize
            512,                // TileP
            64,                 // TileQ
            128,                // VecP
            4,                  // VecQ
            16                  // VecB
                > (meta.tile_M, // GemmM
                   meta.K,      // GemmK
                   meta.N,      // GemmN
                   wp_ptr,      // w_bits
                   lut_ptr,     // wdeq_lut
                   out_ptr      // w_hf
                  );

        if (status != 0)
        {
            std::cerr << "Error in hvx_dequantize_weights: " << status << std::endl;
            return 1;
        }
    }

    std::cout << "\nDequantization complete!" << std::endl;

    // Compare results
    std::cout << "\nComparing results..." << std::endl;
    bool passed = compare_fp16_arrays(w_hf.data(), w_deq_ref.data(),
                                      meta.M * meta.K, 1e-3, 1e-5);

    if (passed)
    {
        std::cout << "\n✓ TEST PASSED - Results match reference!" << std::endl;
    }
    else
    {
        std::cout << "\n✗ TEST FAILED - Results do not match!" << std::endl;

        // Save output for debugging
        std::ofstream out_file("w_hf_output.bin", std::ios::binary);
        out_file.write(reinterpret_cast<const char *>(w_hf.data()),
                       w_hf.size() * sizeof(__fp16));
        std::cout << "Output saved to w_hf_output.bin for debugging" << std::endl;
    }

    return passed ? 0 : 1;
}