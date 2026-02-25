#include <cassert>
#include <iostream>
#include <fstream>
#include <random>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnTypes.h"
#include "qnn_device.h"
#include "qnn_dynload.h"
#include "qnn_backend.h"
#include "qnn_context.h"
#include "qnn_graph.h"
#include "qnn_profiler.h"
#include "qnn_sharedbuffer.h"
#include "qnn_tensor.h"
#include "qnn_backendcache.h"
#include "qnn_mem_manager.h"
#include "qnn_log.h"
#include "precompute_ref.h"
#include "tman_linear_ref.h"
#include "tman_finalize_ref.h"

template <typename T>
static bool load_raw(const std::string& path, std::vector<T>& out, size_t numel) {
  out.resize(numel);
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Failed to open for read: " << path << "\n";
    return false;
  }
  in.read(reinterpret_cast<char*>(out.data()), sizeof(T) * numel);
  if (!in.good()) {
    std::cerr << "Read failed or file too small: " << path << "\n";
    return false;
  }
  return true;
}

static bool load_f32_raw(const std::string& path, std::vector<float>& out, size_t numel) {
  out.resize(numel);
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Failed to open for read: " << path << "\n";
    return false;
  }

  in.read(reinterpret_cast<char*>(out.data()), sizeof(float) * numel);
  if (!in.good()) {
    std::cerr << "Read failed or file too small: " << path << "\n";
    return false;
  }

  // 파일이 더 큰 경우는 허용(원하면 체크 가능)
  return true;
}

// reference cpu code
// A: [B, M, K]
// B: if (!transposeB) [BB, K, N]
//    if ( transposeB) [BB, N, K]  (we use B^T in multiplication)
//    BB must be 1 or B
// Out: [B, M, N]
static void batch_matmul_f32(
    const float* A,
    const float* Bm,
    float* Out,
    int B, int M, int K, int N, int BB,
    bool transposeB)
{
  // zero init
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        Out[((b*M + i)*N) + j] = 0.0f;
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    const float* Ab = A  + (size_t)b * M * K;
    const float* Bb;
    if(BB == 1){
        Bb = Bm;
    } else{
        // BB == B
        Bb = Bm + (size_t)b * (transposeB ? (size_t)N*K : (size_t)K*N);
    }
    
    float* Ob       = Out+ (size_t)b * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          const float a = Ab[(size_t)i*K + k];

          // 핵심: transposeB면 B의 "원본" shape이 [N, K]
          // 곱셈에서는 B^T를 쓰므로, B^T의 (k, j)는 원본 B의 (j, k)
          const float bval = transposeB
              ? Bb[(size_t)j*K + k]   // B_original[j, k]
              : Bb[(size_t)k*N + j];  // B_original[k, j]

          acc += a * bval;
        }
        Ob[(size_t)i*N + j] = acc;
      }
    }
  }
}

struct RunResult{
    std::vector<void*> input_ptrs;
    std::vector<Qnn_MemHandle_t> input_handles;
    std::vector<Qnn_Tensor_t> input_metas;
    std::vector<Qnn_Tensor_t> output_metas;
    std::vector<std::vector<uint8_t>> output_bufs;
};

static bool RunOneGraph(
    const std::string& graph_name,
    const QnnInterface_t* be,
    Qnn_GraphHandle_t graph_handle,
    HtpBackendCacheRuntime& backendcache,
    QnnMemManagerRuntime& mem,
    SharedBuffer& sb,
    SharedBuffer::Arena& arena,
    Qnn_ProfileHandle_t ph,
    RunResult& rr
){
    rr.input_metas = backendcache.GetGraphInputs(graph_name);
    rr.output_metas = backendcache.GetGraphOutputs(graph_name);
        
    std::cout << "graph_name=" << graph_name
              << " num_inputs=" << rr.input_metas.size()
              << " num_outputs=" << rr.output_metas.size() << "\n";

    if (rr.input_metas.empty() || rr.output_metas.empty()) {
        std::cerr << "[QNN] empty graph IO meta. check graph name or backendcache parsing\n";
        return false;
    }

    // dtype size helper (필요한 것만)
    auto dtype_size = [](Qnn_DataType_t dt) -> size_t {
        switch (dt) {
            case QNN_DATATYPE_FLOAT_32: return 4;
            case QNN_DATATYPE_FLOAT_16: return 2;
            case QNN_DATATYPE_UINT_8:
            case QNN_DATATYPE_INT_8:
            case QNN_DATATYPE_BOOL_8:
            case QNN_DATATYPE_SFIXED_POINT_8:
            case QNN_DATATYPE_UFIXED_POINT_8: return 1;
            case QNN_DATATYPE_INT_16:
            case QNN_DATATYPE_UINT_16:
            case QNN_DATATYPE_SFIXED_POINT_16:
            case QNN_DATATYPE_UFIXED_POINT_16: return 2;
            case QNN_DATATYPE_INT_32:
            case QNN_DATATYPE_UINT_32: return 4;
            case QNN_DATATYPE_INT_64:
            case QNN_DATATYPE_UINT_64: return 8;
            default: return 0;
        }
    };

    auto calc_bytes_from_meta = [&](const Qnn_Tensor_t& t) -> size_t {
        auto* tv = QNN_TENSOR_VER_PTR(t);
        size_t bytes = dtype_size(tv->dataType);
        for (uint32_t i = 0; i < tv->rank; ++i) bytes *= tv->dimensions[i];
        return bytes;
    };

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    // input buffer
    rr.input_ptrs.assign(rr.input_metas.size(), nullptr);
    rr.input_handles.assign(rr.input_metas.size(), nullptr);


    for (size_t i = 0; i < rr.input_metas.size(); ++i) {
        auto* tv = QNN_TENSOR_VER_PTR(rr.input_metas[i]);
        size_t bytes = tv->clientBuf.dataSize ? tv->clientBuf.dataSize : calc_bytes_from_meta(rr.input_metas[i]);
        
        void* ptr = nullptr;
        Qnn_MemHandle_t h = nullptr;
        if(!mem.RegisterTensorInSharedArena(sb, arena, rr.input_metas[i], bytes, 64, &ptr, &h)){
            std::cerr << "RegisterTensorInSharedArena failed\n";
            return -1;
        }

        // 지금은 float32 입력만 랜덤으로 채우자 (네 모델이 fp32면 OK)
        if (tv->dataType == QNN_DATATYPE_FLOAT_32) {
            float* p = reinterpret_cast<float*>(ptr);
            size_t n = bytes / sizeof(float);
            // Debug: only first group (4 values) non-zero, rest zero
            for (size_t k = 0; k < n; ++k) p[k] = 0.0f;
            p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
        } else {
            // 다른 dtype은 일단 0으로
            std::cerr << "Should not reach here\n";
        }

        rr.input_ptrs[i] = ptr;
        rr.input_handles[i] = h;

        std::cout << "Input[" << i << "] name=" << tv->name
                  << " bytes=" << bytes
                  << " dtype=" << tv->dataType
                  << " rank=" << tv->rank << "\n";
    }
    rr.output_bufs.resize(rr.output_metas.size());

    for (size_t i = 0; i < rr.output_metas.size(); ++i) {
        auto* tv = QNN_TENSOR_VER_PTR(rr.output_metas[i]);
        size_t bytes = tv->clientBuf.dataSize ? tv->clientBuf.dataSize : calc_bytes_from_meta(rr.output_metas[i]);
        rr.output_bufs[i].resize(bytes);
        std::memset(rr.output_bufs[i].data(), 0, bytes);

        tv->memType = QNN_TENSORMEMTYPE_RAW;
        tv->clientBuf.data = rr.output_bufs[i].data();
        tv->clientBuf.dataSize = bytes;

        std::cout << "Output[" << i << "] name=" << tv->name
                  << " bytes=" << bytes
                  << " dtype=" << tv->dataType
                  << " rank=" << tv->rank << "\n";
    }

    auto& api = be->QNN_INTERFACE_VER_NAME;

    Qnn_ErrorHandle_t err = api.graphExecute(
        graph_handle,
        rr.input_metas.data(),
        static_cast<uint32_t>(rr.input_metas.size()),
        rr.output_metas.data(),
        static_cast<uint32_t>(rr.output_metas.size()),
        /*profile=*/ph,
        /*signal=*/nullptr);

    if (err != QNN_SUCCESS) {
        std::cerr << "[QNN] graphExecute failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
        return false;
    }
    std::cout << "GRAPH EXECUTE: " << graph_name << "\n";

    return true;
}

static float fp16_to_fp32(uint16_t h) {
    uint16_t sign = (h & 0x8000u) >> 15;
    uint16_t exp  = (h & 0x7C00u) >> 10;
    uint16_t frac = (h & 0x03FFu);

    if (exp == 0) {
        if (frac == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // subnormal
        return (sign ? -1.0f : 1.0f) *
               std::ldexp(static_cast<float>(frac), -24);
    }
    else if (exp == 31) {
        if (frac == 0)
            return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    // normal
    float mant = 1.0f + static_cast<float>(frac) / 1024.0f;
    int e = static_cast<int>(exp) - 15;
    float val = std::ldexp(mant, e);

    return sign ? -val : val;
}

static void DumpOutputs(
    const std::vector<Qnn_Tensor_t>& output_metas,
    const std::vector<std::vector<uint8_t>>& output_bufs,
    size_t max_f32 = 16,
    size_t max_hex = 64
){
    // ===== output dump (float32 기준으로 몇 개만) =====
    for (size_t i = 0; i < output_metas.size(); ++i) {
        auto* tv = QNN_TENSOR_VER_PTR(output_metas[i]);
        std::cout << "=== Output[" << i << "] " << tv->name << " ===\n";
        std::cout << "SIBAL : " << tv->dataType << std::endl;
        if (tv->dataType == QNN_DATATYPE_FLOAT_32) {
            const float* p = reinterpret_cast<const float*>(output_bufs[i].data());
            size_t n = output_bufs[i].size() / sizeof(float);
            size_t show = std::min<size_t>(n, 16);
            for (size_t k = 0; k < show; ++k) {
                std::cout << p[k] << (k + 1 == show ? "\n" : ", ");
            }
        }
        else if (tv->dataType == QNN_DATATYPE_FLOAT_16) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(output_bufs[i].data());
            
            std::cout << "BITS EXPRESSION : " << std::endl;
            uint16_t v = p[0];

            // 16비트 MSB → LSB 순서로 출력
            for (int b = 15; b >= 0; --b) {
                std::cout << ((v >> b) & 1);
                if (b == 15 || b == 10) std::cout << " ";  
                // fp16 구조 보기 좋게:
                // [sign] [exponent(5)] [mantissa(10)]
            }
            std::cout << std::endl;
            
            size_t n = output_bufs[i].size() / sizeof(uint16_t);
            std::cout << "number of elements is " << n << std::endl;
            size_t show = std::min<size_t>(n, 16);
            for (size_t k = 0; k < show; ++k) {
                float f = fp16_to_fp32(p[k]);
                std::cout << f << (k + 1 == show ? "\n" : ", ");
            }
        } else {
            // 다른 dtype이면 raw hex로 앞부분만
            size_t show = std::min<size_t>(output_bufs[i].size(), 64);
            for (size_t k = 0; k < show; ++k) {
                printf("%02x%s", output_bufs[i][k], ((k + 1) % 16 == 0) ? "\n" : " ");
            }
            if (show % 16 != 0) printf("\n");
        }
    }

}

struct CpuRefOut {
  std::vector<float> out;   // [B*L*D]
};


static bool ComputeCpuReference(
    bool is_kv,
    const void* x_ptr,   // input_ptrs[0]
    const void* y_ptr,   // input_ptrs[1] (prefill에서만 사용, kv면 무시 가능)
    unsigned int B, unsigned int L, unsigned int D, unsigned int C,
    CpuRefOut& ref
) {
  // load static weights
  std::vector<float> static_q, static_k, static_v;
  if (!load_f32_raw("static_q.bin", static_q, (size_t)D * C)) return false;
  if (!load_f32_raw("static_k.bin", static_k, (size_t)D * C)) return false;
//   if (!load_f32_raw("static_v.bin", static_v, (size_t)D * C)) return false;

  std::vector<float> q, k, v, attn;
  q.resize((size_t)B * L * D);
  k.resize((size_t)B * L * D);
  v.resize((size_t)B * L * D);
  attn.resize((size_t)B * L * L);
  ref.out.resize((size_t)B * L * D);

  batch_matmul_f32(
      reinterpret_cast<const float*>(x_ptr),
      static_cast<const float*>(static_q.data()),
      q.data(), B, L, C, D, 1, true);

  std::cout << "시발 q중앙값 : " << q[0] << ", " << q[1] << ", " << q[2] << "\n";

  batch_matmul_f32(
      reinterpret_cast<const float*>(x_ptr),
      static_cast<const float*>(static_k.data()),
      k.data(), B, L, C, D, 1, true);
  std::cout << "시발 k중앙값 : " << k[0] << ", " << k[1] << ", " << k[2] << "\n";

//   std::vector<float> dequantw; // shape (C, D)
//   if(!load_f32_raw("w_dequant.bin", dequantw, (size_t)C * D)) return false;

//   batch_matmul_f32(
//     reinterpret_cast<const float*>(x_ptr),
//     dequantw.data(),
//     v.data(), B, L, C, D, 1, false
//   );
//   std::cout << "시발 v중앙값 : " << v[0] << ", " << v[1] << ", " << v[2] << "\n";
  
  batch_matmul_f32(
      static_cast<const float*>(q.data()),
      static_cast<const float*>(k.data()),
      attn.data(), B, L, D, L, B, true);
  std::cout << "시발 attn중앙값 : " << attn[0] << ", " << attn[1] << ", " << attn[2] << "\n";

  batch_matmul_f32(
      static_cast<const float*>(attn.data()),
      static_cast<const float*>(v.data()),
      ref.out.data(), B, L, L, D, B, false);
  std::cout << "시발 ref_out중앙값 : " << ref.out[0] << ", " << ref.out[1] << ", " << ref.out[2] << "\n";

  return true;
}

static void DumpCpuReferenceHead(
    const CpuRefOut& ref,
    const char* tag,
    size_t max_f32 = 16
) {
  std::cout << "====== CPU REFERENCE OUTPUT (" << tag << ") ======\n";
  size_t n = ref.out.size();
  size_t show = std::min<size_t>(n, max_f32);
  for (size_t k = 0; k < show; ++k) {
    std::cout << ref.out[k] << (k + 1 == show ? "\n" : ", ");
  }
}

static void DumpQnnOutputHead(
    const std::vector<Qnn_Tensor_t>& output_metas,
    const std::vector<std::vector<uint8_t>>& output_bufs,
    const char* tag,
    size_t max_f32 = 16
) {
  std::cout << "====== QNN OUTPUT (" << tag << ") ======\n";
  if (output_bufs.empty()) {
    std::cout << "(no outputs)\n";
    return;
  }
  auto* tv = QNN_TENSOR_VER_PTR(output_metas[0]);
  if (tv->dataType == QNN_DATATYPE_FLOAT_32) {
    const float* p = reinterpret_cast<const float*>(output_bufs[0].data());
    size_t n = output_bufs[0].size() / sizeof(float);
    size_t show = std::min<size_t>(n, max_f32);
    for (size_t k = 0; k < show; ++k) {
        std::cout << p[k] << (k + 1 == show ? "\n" : ", ");
    }
  } else if (tv->dataType == QNN_DATATYPE_FLOAT_16){
    const uint16_t* p = reinterpret_cast<const uint16_t*>(output_bufs[0].data());
    size_t n = output_bufs[0].size() / sizeof(uint16_t);
    size_t show = std::min<size_t>(n, max_f32);
    for (size_t k = 0; k < show; ++k) {
        float f = fp16_to_fp32(p[k]);
        std::cout << f << (k + 1 == show ? "\n" : ", ");
    }
  } else if (tv->dataType == QNN_DATATYPE_UINT_8){
    const uint8_t* p = reinterpret_cast<const uint8_t*>(output_bufs[0].data());
    size_t n = output_bufs[0].size() / sizeof(uint8_t);
    size_t show = std::min<size_t>(n, max_f32);
    for (size_t k = 0; k < show; ++k) {
        std::cout << static_cast<unsigned int>(p[k]) << (k + 1 == show ? "\n" : ", ");
    }
  } else {
    std::cout << "(unsupported dtype for head dump)\n";
  }
}

static void ComparePrecomputeRef(
    const void* x_fp32_ptr,      // fp32 input activations
    const uint8_t* qnn_output,   // raw QNN output buffer
    size_t qnn_output_bytes,
    int32_t gemm_k,              // = C = 2048
    int32_t group_size           // = 128
)
{
    int32_t ref_size = precompute_ref_bufsize(gemm_k, group_size);
    std::cout << "=== Precompute CPU Reference Comparison ===\n";
    std::cout << "  gemm_k=" << gemm_k << " group_size=" << group_size
              << " ref_bufsize=" << ref_size << " qnn_bufsize=" << qnn_output_bytes << "\n";

    std::vector<uint8_t> ref_buf(ref_size, 0);
    precompute_ref(gemm_k, group_size,
                   reinterpret_cast<const float*>(x_fp32_ptr),
                   ref_buf.data());

    // Layout offsets
    constexpr int32_t g = 4, lut_size = 16;
    const int32_t Q = gemm_k / g;
    const int32_t l_count = Q * lut_size;
    const int32_t l_bytes = l_count * (int32_t)sizeof(int16_t);
    const int32_t num_scale = gemm_k / 256;
    const int32_t ls_pad = std::max(num_scale, (int32_t)(128 / (int32_t)sizeof(float)));
    const int32_t ls_bytes = ls_pad * (int32_t)sizeof(float);

    const int16_t* ref_l  = reinterpret_cast<const int16_t*>(ref_buf.data());
    const float*   ref_ls = reinterpret_cast<const float*>(ref_buf.data() + l_bytes);
    const float*   ref_lb = reinterpret_cast<const float*>(ref_buf.data() + l_bytes + ls_bytes);

    const int16_t* qnn_l  = reinterpret_cast<const int16_t*>(qnn_output);
    const float*   qnn_ls = reinterpret_cast<const float*>(qnn_output + l_bytes);
    const float*   qnn_lb = reinterpret_cast<const float*>(qnn_output + l_bytes + ls_bytes);

    // Compare ls (scales)
    std::cout << "\n  [ls] scales (" << num_scale << " values):\n";
    for (int32_t i = 0; i < num_scale; i++) {
        float diff = fabsf(ref_ls[i] - qnn_ls[i]);
        const char* mark = (diff > 1e-4f) ? " <-- MISMATCH" : "";
        std::cout << "    ls[" << i << "] ref=" << ref_ls[i]
                  << " qnn=" << qnn_ls[i] << " diff=" << diff << mark << "\n";
    }

    // Compare lb (biases)
    int32_t num_bias = Q / (group_size / g);
    std::cout << "\n  [lb] biases (" << num_bias << " values):\n";
    for (int32_t i = 0; i < num_bias; i++) {
        float diff = fabsf(ref_lb[i] - qnn_lb[i]);
        const char* mark = (diff > 1e-2f) ? " <-- MISMATCH" : "";
        std::cout << "    lb[" << i << "] ref=" << ref_lb[i]
                  << " qnn=" << qnn_lb[i] << " diff=" << diff << mark << "\n";
    }

    // Compare LUT (int16) - summary
    int32_t match = 0, total = l_count;
    int32_t max_diff = 0;
    for (int32_t i = 0; i < total; i++) {
        int32_t d = std::abs((int32_t)ref_l[i] - (int32_t)qnn_l[i]);
        if (d == 0) match++;
        if (d > max_diff) max_diff = d;
    }
    std::cout << "\n  [LUT] int16 entries: " << match << "/" << total << " exact match"
              << ", max_diff=" << max_diff << "\n";

    // Show first few mismatches
    int shown = 0;
    for (int32_t i = 0; i < total && shown < 16; i++) {
        if (ref_l[i] != qnn_l[i]) {
            std::cout << "    l[" << i << "] ref=" << ref_l[i]
                      << " qnn=" << qnn_l[i] << "\n";
            shown++;
        }
    }
    if (shown == 0)
        std::cout << "    (all LUT entries match exactly)\n";

    std::cout << "=== End Precompute Comparison ===\n";
}

// Compare TMANLinear CPU reference output with QNN c_tns output.
//
// Internally runs precompute_ref() to produce the l/ls/lb buffer
// (since l_tns is now NATIVE and not a graph output), then feeds
// that into tman_linear_ref() along with unpacked weights/scales.
//
// QNN output (c_tns) is UINT_8 raw bytes but contains float data:
//   M * bits floats = 2048 * 4 = 8192 floats = 32768 bytes.
static void CompareLinearRef(
    const void* x_fp32_ptr,      // fp32 input activations, length K
    const uint8_t* qnn_output,   // raw QNN c_tns output buffer
    size_t qnn_output_bytes,
    int32_t gemm_m,              // = D = 2048
    int32_t gemm_k,              // = C = 8192
    int32_t bits,                // = 4
    int32_t group_size           // = 128
)
{
    std::cout << "=== TMANLinear CPU Reference Comparison ===\n";
    std::cout << "  gemm_m=" << gemm_m << " gemm_k=" << gemm_k
              << " bits=" << bits << " group_size=" << group_size << "\n";

    // Step 1: Run precompute_ref to produce l/ls/lb buffer
    int32_t precompute_size = precompute_ref_bufsize(gemm_k, group_size);
    std::vector<uint8_t> precompute_buf(precompute_size, 0);
    precompute_ref(gemm_k, group_size,
                   reinterpret_cast<const float*>(x_fp32_ptr),
                   precompute_buf.data());
    std::cout << "  precompute_ref done, bufsize=" << precompute_size << "\n";

    // Step 2: Load unpacked weights
    //   uint8, shape (M, K) = (2048, 8192) = 16,777,216 bytes
    const int32_t M = gemm_m;
    const int32_t K = gemm_k;
    std::vector<uint8_t> w_unpacked;
    if (!load_raw<uint8_t>("w_unpacked.bin", w_unpacked, (size_t)M * K)) {
        std::cerr << "  Failed to load w_unpacked.bin\n";
        return;
    }
    std::cout << "  w_unpacked loaded: " << w_unpacked.size() << " bytes\n";

    // Step 3: Load unpacked scales
    //   fp16 as uint16, shape (M, K/group_size) = (2048, 64) = 131,072 entries
    const int32_t num_wgt_groups = K / group_size;  // 8192 / 128 = 64
    std::vector<uint16_t> s_unpacked;
    if (!load_raw<uint16_t>("s_unpacked.bin", s_unpacked, (size_t)M * num_wgt_groups)) {
        std::cerr << "  Failed to load s_unpacked.bin\n";
        return;
    }
    std::cout << "  s_unpacked loaded: " << s_unpacked.size() << " uint16 entries\n";

    // Step 4: Run CPU reference
    //   Output: M * bits = 2048 * 4 = 8192 floats = 32768 bytes
    int32_t ref_bufsize = tman_linear_ref_bufsize(gemm_m, bits);
    std::vector<float> ref_output(gemm_m * bits, 0.0f);
    tman_linear_ref(gemm_m, gemm_k, bits, group_size,
                    precompute_buf.data(),
                    w_unpacked.data(),
                    s_unpacked.data(),
                    ref_output.data());

    // Step 5: Compare with QNN output
    const int32_t num_floats = gemm_m * bits;  // 2048 * 4 = 8192
    const int32_t expected_bytes = num_floats * (int32_t)sizeof(float);  // 32768
    std::cout << "  ref_bufsize=" << ref_bufsize << " bytes"
              << " qnn_output_bytes=" << qnn_output_bytes
              << " expected=" << expected_bytes << " bytes\n";

    if ((int32_t)qnn_output_bytes < expected_bytes) {
        std::cerr << "  QNN output buffer too small!\n";
        return;
    }

    const float* qnn_floats = reinterpret_cast<const float*>(qnn_output);

    int32_t exact_match = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int32_t max_abs_idx = 0;
    double sum_abs_diff = 0.0;

    for (int32_t i = 0; i < num_floats; i++) {
        float ref_val = ref_output[i];
        float qnn_val = qnn_floats[i];
        float abs_diff = fabsf(ref_val - qnn_val);
        sum_abs_diff += abs_diff;

        if (abs_diff == 0.0f) exact_match++;

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_abs_idx = i;
        }

        float denom = std::max(fabsf(ref_val), fabsf(qnn_val));
        if (denom > 1e-8f) {
            float rel = abs_diff / denom;
            if (rel > max_rel_diff) max_rel_diff = rel;
        }
    }

    std::cout << "\n  [Result] " << exact_match << "/" << num_floats << " exact match ("
              << (100.0f * exact_match / num_floats) << "%)\n";
    std::cout << "  max_abs_diff=" << max_abs_diff << " at index " << max_abs_idx
              << " (ref=" << ref_output[max_abs_idx] << " qnn=" << qnn_floats[max_abs_idx] << ")\n";
    std::cout << "  max_rel_diff=" << max_rel_diff << "\n";
    std::cout << "  mean_abs_diff=" << (sum_abs_diff / num_floats) << "\n";

    // Show first 16 values
    int32_t show = std::min(num_floats, (int32_t)16);
    std::cout << "\n  First " << show << " values:\n";
    for (int32_t i = 0; i < show; i++) {
        float diff = fabsf(ref_output[i] - qnn_floats[i]);
        std::cout << "    [" << i << "] ref=" << ref_output[i]
                  << " qnn=" << qnn_floats[i] << " diff=" << diff << "\n";
    }

    // Show first mismatches (abs_diff > 1e-3)
    int32_t mismatch_shown = 0;
    std::cout << "\n  First mismatches (abs_diff > 1e-3):\n";
    for (int32_t i = 0; i < num_floats && mismatch_shown < 16; i++) {
        float diff = fabsf(ref_output[i] - qnn_floats[i]);
        if (diff > 1e-3f) {
            std::cout << "    [" << i << "] ref=" << ref_output[i]
                      << " qnn=" << qnn_floats[i] << " diff=" << diff << "\n";
            mismatch_shown++;
        }
    }
    if (mismatch_shown == 0)
        std::cout << "    (all within 1e-3 tolerance)\n";

    std::cout << "=== End TMANLinear Comparison ===\n";
}

// Compare TMANFinalize CPU reference output with QNN y_tns output.
//
// Runs the full chain internally:
//   precompute_ref → tman_linear_ref → tman_finalize_ref
// Then compares the finalize fp16 output with QNN y_tns (fp16).
static void CompareFinalizeRef(
    const void* x_fp32_ptr,      // fp32 input activations, length K
    const uint8_t* qnn_output,   // raw QNN y_tns output buffer (fp16)
    size_t qnn_output_bytes,
    int32_t gemm_m,              // = D = 2048
    int32_t gemm_k,              // = C = 8192
    int32_t bits,                // = 4
    int32_t group_size           // = 128
)
{
    std::cout << "=== TMANFinalize CPU Reference Comparison ===\n";
    std::cout << "  gemm_m=" << gemm_m << " gemm_k=" << gemm_k
              << " bits=" << bits << " group_size=" << group_size << "\n";

    // Step 1: precompute_ref → l/ls/lb buffer
    int32_t precompute_size = precompute_ref_bufsize(gemm_k, group_size);
    std::vector<uint8_t> precompute_buf(precompute_size, 0);
    precompute_ref(gemm_k, group_size,
                   reinterpret_cast<const float*>(x_fp32_ptr),
                   precompute_buf.data());
    std::cout << "  precompute_ref done, bufsize=" << precompute_size << "\n";

    // Step 2: Load unpacked weights and scales
    const int32_t M = gemm_m;
    const int32_t K = gemm_k;
    const int32_t num_wgt_groups = K / group_size;

    std::vector<uint8_t> w_unpacked;
    if (!load_raw<uint8_t>("w_unpacked.bin", w_unpacked, (size_t)M * K)) {
        std::cerr << "  Failed to load w_unpacked.bin\n";
        return;
    }
    std::vector<uint16_t> s_unpacked;
    if (!load_raw<uint16_t>("s_unpacked.bin", s_unpacked, (size_t)M * num_wgt_groups)) {
        std::cerr << "  Failed to load s_unpacked.bin\n";
        return;
    }
    std::cout << "  weights/scales loaded\n";

    // Step 3: tman_linear_ref → float output (interleaved layout)
    std::vector<float> linear_output(M * bits, 0.0f);
    tman_linear_ref(gemm_m, gemm_k, bits, group_size,
                    precompute_buf.data(),
                    w_unpacked.data(),
                    s_unpacked.data(),
                    linear_output.data());

    // Step 4: tman_finalize_ref → fp16 output (natural channel order)
    std::vector<uint16_t> ref_fp16(M);
    tman_finalize_ref(gemm_m, bits, linear_output.data(), ref_fp16.data());

    // Step 5: Compare with QNN output (fp16)
    const int32_t expected_bytes = M * (int32_t)sizeof(uint16_t);  // 4096
    std::cout << "  qnn_output_bytes=" << qnn_output_bytes
              << " expected=" << expected_bytes << " bytes\n";

    if ((int32_t)qnn_output_bytes < expected_bytes) {
        std::cerr << "  QNN output buffer too small!\n";
        return;
    }

    const uint16_t* qnn_fp16 = reinterpret_cast<const uint16_t*>(qnn_output);

    int32_t exact_match = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int32_t max_abs_idx = 0;
    double sum_abs_diff = 0.0;

    for (int32_t i = 0; i < M; i++) {
        float ref_val = tman_fp16_to_f32(ref_fp16[i]);
        float qnn_val = tman_fp16_to_f32(qnn_fp16[i]);
        float abs_diff = fabsf(ref_val - qnn_val);
        sum_abs_diff += abs_diff;

        if (ref_fp16[i] == qnn_fp16[i]) exact_match++;

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_abs_idx = i;
        }

        float denom = std::max(fabsf(ref_val), fabsf(qnn_val));
        if (denom > 1e-8f) {
            float rel = abs_diff / denom;
            if (rel > max_rel_diff) max_rel_diff = rel;
        }
    }

    std::cout << "\n  [Result] " << exact_match << "/" << M << " exact match ("
              << (100.0f * exact_match / M) << "%)\n";
    std::cout << "  max_abs_diff=" << max_abs_diff << " at index " << max_abs_idx
              << " (ref=" << tman_fp16_to_f32(ref_fp16[max_abs_idx])
              << " qnn=" << tman_fp16_to_f32(qnn_fp16[max_abs_idx]) << ")\n";
    std::cout << "  max_rel_diff=" << max_rel_diff << "\n";
    std::cout << "  mean_abs_diff=" << (sum_abs_diff / M) << "\n";

    int32_t show = std::min(M, (int32_t)16);
    std::cout << "\n  First " << show << " values:\n";
    for (int32_t i = 0; i < show; i++) {
        float ref_val = tman_fp16_to_f32(ref_fp16[i]);
        float qnn_val = tman_fp16_to_f32(qnn_fp16[i]);
        std::cout << "    [" << i << "] ref=" << ref_val << " qnn=" << qnn_val
                  << " diff=" << fabsf(ref_val - qnn_val) << "\n";
    }

    int32_t mismatch_shown = 0;
    std::cout << "\n  First mismatches (abs_diff > 1e-2):\n";
    for (int32_t i = 0; i < M && mismatch_shown < 16; i++) {
        float ref_val = tman_fp16_to_f32(ref_fp16[i]);
        float qnn_val = tman_fp16_to_f32(qnn_fp16[i]);
        float diff = fabsf(ref_val - qnn_val);
        if (diff > 1e-2f) {
            std::cout << "    [" << i << "] ref=" << ref_val
                      << " qnn=" << qnn_val << " diff=" << diff << "\n";
            mismatch_shown++;
        }
    }
    if (mismatch_shown == 0)
        std::cout << "    (all within 1e-2 tolerance)\n";

    std::cout << "=== End TMANFinalize Comparison ===\n";
}

static void DumpAndSerializeProfiler(
    QnnProfilerRuntime& profiler,
    const std::string& graph_name
) {
  profiler.DumpEventsRecursive(/*dump_sub_events=*/true, /*max_depth=*/32);

  if (!profiler.SerializeAfterExecute(graph_name.c_str())) {
    std::cerr << "[QNN] SerializeAfterExecute failed for " << graph_name << "\n";
  }
}

static bool PostProcessOneGraphRun(
    const std::string& graph_name,
    bool is_kv,
    const std::vector<void*>& input_ptrs,  // input_ptrs[0]=x, input_ptrs[1]=y (prefill)
    const std::vector<Qnn_Tensor_t>& output_metas,
    const std::vector<std::vector<uint8_t>>& output_bufs,
    QnnProfilerRuntime& profiler
) {
  // 1) output dump
  DumpOutputs(output_metas, output_bufs, /*max_f32=*/16, /*max_hex=*/64);

  // 2) profiler dump + serialize
  DumpAndSerializeProfiler(profiler, graph_name);

  // 3) cpu reference
  const unsigned int L = is_kv ? 1 : 30;
  const unsigned int B = 1, D = 2048, C = 8192;
  if (input_ptrs.empty() || input_ptrs[0] == nullptr) {
    std::cerr << "[QNN] input_ptrs[0] missing\n";
    return false;
  }
  if (!is_kv && (input_ptrs.size() < 2 || input_ptrs[1] == nullptr)) {
    std::cerr << "[QNN] prefill needs input_ptrs[1]\n";
    return false;
  }

  CpuRefOut ref;
//   if (!ComputeCpuReference(
//           is_kv,
//           /*x_ptr=*/input_ptrs[0],
//           /*y_ptr=*/(is_kv ? nullptr : input_ptrs[1]),
//           B, L, D, C,
//           ref)) {
//     std::cerr << "[QNN] ComputeCpuReference failed for " << graph_name << "\n";
//     return false;
//   }

  DumpQnnOutputHead(output_metas, output_bufs, graph_name.c_str(), /*max_f32=*/16);
  DumpCpuReferenceHead(ref, graph_name.c_str(), /*max_f32=*/16);

  std::cout << "IS KV? " << (is_kv ? "YES" : "NO") << "\n";

  if (is_kv) {
    // c_tns is now NATIVE (intermediate), output_bufs[0] = y_tns (TMANFinalize output)
    // y_tns: FLOAT_16, M fp16 values = 2048 * 2 = 4096 bytes
    constexpr int32_t GEMM_M = D;   // 2048
    constexpr int32_t GEMM_K = C;   // 8192
    constexpr int32_t BITS = 4;
    constexpr int32_t GRP_SIZE = 128;
    CompareFinalizeRef(
        input_ptrs[0],            // fp32 activations (length K = 8192 floats)
        output_bufs[0].data(),    // QNN y_tns output (fp16)
        output_bufs[0].size(),
        GEMM_M, GEMM_K, BITS, GRP_SIZE);
  }

  return true;
}

int main(int argc, char** argv){
    std::ifstream bin("multi_graph.bin", std::ios::binary | std::ios::ate);
    assert(bin.is_open());

    size_t binSize = bin.tellg();
    bin.seekg(0);

    std::vector<uint8_t> binData(binSize);
    bin.read(reinterpret_cast<char *>(binData.data()), binSize);
    bin.close();

    printf("Loaded context binary: %zu bytes\n", binSize);

    const std::string backend_so = "libQnnHtp.so";
    const std::string system_so = "libQnnSystem.so";

    auto& qnn = QnnDynLoad::Instance();
    if (!qnn.LoadAll(backend_so, system_so)) {
        std::cerr << "Failed to load QNN backend or system\n";
        return -1;
    }

    std::cout << "QNN backend loaded: backendId= " << qnn.Backend()->backendId << "\n";
    std::cout << "QNN system loaded: systemId= " << qnn.System() << "\n";
    
    Qnn_LogHandle_t logHandle = nullptr;
    if (!CreateQnnLogger(qnn.Backend(), &logHandle, /*QNN_LOG_LEVEL_VERBOSE*/ QNN_LOG_LEVEL_INFO)) {
        std::cerr << "Failed to create QNN logger (continuing without logger)\n";
        return -1;
    } else {
        std::cout << "QNN logger created. logHandle=" << logHandle << "\n";
    }

    QnnBackendRuntime backend;
    if (!backend.Create(qnn.Backend(), /*logger_handler=*/logHandle)){
        std::cerr << "backendCreate failed\n";
        return -1;
    }
    std::cout << "backendCreate OK\n";

    QnnDeviceRuntime device;
    if(!device.Create(qnn.Backend(), /*logger_handler=*/logHandle)){
        std::cerr << "deviceCreate failed\n";
        return -1;
    }
    std::cout << "deviceCreate OK\n";
    
    HtpBackendCacheRuntime backendcache;
    QnnContextBinary blob;
    blob.buffer = binData.data();
    blob.nbytes = static_cast<uint32_t>(binSize);

    if(!backendcache.Create(qnn.System(), blob)){
        std::cerr << "backendcacheCreate failed\n";
        return -1;
    }

    QnnProfilerRuntime profiler;
    if(!profiler.Create(qnn.Backend(), qnn.System(), backend.Handle(), QnnProfileLevel::Optrace, true, "qnn.log")){
        std::cerr << "ProfilerCreate failed\n";
        return -1;
    }

    QnnContextRuntime ctx;
    // ctx.SetMultiContexts(true, /*max_sf_buf_size=*/spill_fill_size);
    if(!ctx.CreateFromBinary(qnn.Backend(), backend.Handle(), device.Handle(), profiler.GetProfiler(), binData.data(), blob.nbytes)){
        std::cerr << "contextCreateFromBinary failed\n";
        return -1;
    }

    const std::string graph_name = "prefill_forward";
    bool is_kv = false;

    QnnGraphRuntime g_prefill, g_kv;
    g_prefill.SetRestoreMode(true);
    g_kv.SetRestoreMode(true);
    // if (!g_prefill.Create(qnn.Backend(), ctx.Handle(), profiler.GetProfiler(), "prefill_forward")) {
    //     std::cerr << "graphCreate for prefill failed\n";
    //     return -1;
    // }

    if (!g_kv.Create(qnn.Backend(), ctx.Handle(), profiler.GetProfiler(), "kv_forward")) {
        std::cerr << "graphCreate for kv failed\n";
        return -1;
    }

    // std::cout << "graphCreate OK. graph_handle for prefill=" << g_prefill.Handle() << " for kv= " << g_kv.Handle() << "\n";

    QnnMemManagerRuntime mem;
    mem.Init(qnn.Backend(), &ctx);

    // ===== 4) host-side buffers 준비 (random input) =====
    auto & sb = SharedBuffer::Instance();
    SharedBuffer::Arena arena;
    // 대충 크게 alloc
    if (!sb.ArenaCreate(arena, 20000000, 64)){
        std::cerr << "ArenaCreate failed\n";
        return -1;
    }

    auto tensor_bytes = [](const Qnn_Tensor_t& t) -> size_t {
        // 보통 metadata에 clientBuf.dataSize가 들어있음
        // 없으면 dims * dtype size로 계산해야 함
        size_t n = QNN_TENSOR_VER_PTR(t)->clientBuf.dataSize;
        return n;
    };

    RunResult rr_prefill, rr_kv;

    // Preregister TODO - memRegister on runtime for now
    // if(!RunOneGraph("prefill_forward", qnn.Backend(), g_prefill.Handle(), backendcache, mem, sb, arena, profiler.GetProfiler(), rr_prefill)){
    //     std::cerr << "Run prefill failed\n";
    //     return -1;
    // }
    if(!RunOneGraph("kv_forward", qnn.Backend(), g_kv.Handle(), backendcache, mem, sb, arena, profiler.GetProfiler(), rr_kv)){
        std::cerr << "Run kv failed\n";
        return -1;
    }


    // ===== 5) execute =====
    // profiler.DumpEvents();
    // std::cout << "DUMP DONE\n";
    profiler.DumpEventsRecursive(/*dump_sub_events=*/true, /*max_depth=*/32);

    if(!profiler.SerializeAfterExecute(graph_name.c_str())){
        std::cerr << "[QNN] SerializeAfterExecute failed\n";
    }

    // if(!PostProcessOneGraphRun("prefill_forward", false, rr_prefill.input_ptrs,
    //         rr_prefill.output_metas, rr_prefill.output_bufs, profiler)){
    //     return -1;
    // }

    if(!PostProcessOneGraphRun("kv_forward", true, rr_kv.input_ptrs,
            rr_kv.output_metas, rr_kv.output_bufs, profiler)){
        return -1;
    }

    std::cout << "[QNN] Releasing resources...\n";
    sb.ArenaDestroy(arena);

    std::cout << "[QNN] Done.\n";
    return 0;

}