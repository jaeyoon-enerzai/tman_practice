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

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

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
            for (size_t k = 0; k < n; ++k) p[k] = dist(rng);
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

        if (tv->dataType == QNN_DATATYPE_FLOAT_32) {
            const float* p = reinterpret_cast<const float*>(output_bufs[i].data());
            size_t n = output_bufs[i].size() / sizeof(float);
            size_t show = std::min<size_t>(n, 16);
            for (size_t k = 0; k < show; ++k) {
                std::cout << p[k] << (k + 1 == show ? "\n" : ", ");
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
  if (!load_f32_raw("static_v.bin", static_v, (size_t)D * C)) return false;

  std::vector<float> wv, q, k, v, attn;
  wv.resize((size_t)D * C);
  q.resize((size_t)B * L * D);
  k.resize((size_t)B * L * D);
  v.resize((size_t)B * L * D);
  attn.resize((size_t)B * L * L);
  ref.out.resize((size_t)B * L * D);

  batch_matmul_f32(
      reinterpret_cast<const float*>(x_ptr),
      static_cast<const float*>(static_q.data()),
      q.data(), B, L, C, D, 1, true);

  batch_matmul_f32(
      reinterpret_cast<const float*>(x_ptr),
      static_cast<const float*>(static_k.data()),
      k.data(), B, L, C, D, 1, true);

  if (!is_kv) {
    // prefill: wv = static_v @ y, v = x @ wv
    batch_matmul_f32(
        static_cast<const float*>(static_v.data()),
        reinterpret_cast<const float*>(y_ptr),
        wv.data(), 1, D, C, C, 1, false);

    batch_matmul_f32(
        reinterpret_cast<const float*>(x_ptr),
        static_cast<const float*>(wv.data()),
        v.data(), B, L, C, D, 1, true);
  } else {
    // kv: v = x @ static_v
    batch_matmul_f32(
        reinterpret_cast<const float*>(x_ptr),
        static_cast<const float*>(static_v.data()),
        v.data(), B, L, C, D, 1, true);
  }

  batch_matmul_f32(
      static_cast<const float*>(q.data()),
      static_cast<const float*>(k.data()),
      attn.data(), B, L, D, L, B, true);

  batch_matmul_f32(
      static_cast<const float*>(attn.data()),
      static_cast<const float*>(v.data()),
      ref.out.data(), B, L, L, D, B, false);

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
    const std::vector<std::vector<uint8_t>>& output_bufs,
    const char* tag,
    size_t max_f32 = 16
) {
  std::cout << "====== QNN OUTPUT (" << tag << ") ======\n";
  if (output_bufs.empty()) {
    std::cout << "(no outputs)\n";
    return;
  }
  const float* p = reinterpret_cast<const float*>(output_bufs[0].data());
  size_t n = output_bufs[0].size() / sizeof(float);
  size_t show = std::min<size_t>(n, max_f32);
  for (size_t k = 0; k < show; ++k) {
    std::cout << p[k] << (k + 1 == show ? "\n" : ", ");
  }
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
  const unsigned int B = 1, L = 30, D = 1024, C = 2048; // 너 기존 그대로 고정
  if (input_ptrs.empty() || input_ptrs[0] == nullptr) {
    std::cerr << "[QNN] input_ptrs[0] missing\n";
    return false;
  }
  if (!is_kv && (input_ptrs.size() < 2 || input_ptrs[1] == nullptr)) {
    std::cerr << "[QNN] prefill needs input_ptrs[1]\n";
    return false;
  }

  CpuRefOut ref;
  if (!ComputeCpuReference(
          is_kv,
          /*x_ptr=*/input_ptrs[0],
          /*y_ptr=*/(is_kv ? nullptr : input_ptrs[1]),
          B, L, D, C,
          ref)) {
    std::cerr << "[QNN] ComputeCpuReference failed for " << graph_name << "\n";
    return false;
  }

  DumpQnnOutputHead(output_bufs, graph_name.c_str(), /*max_f32=*/16);
  DumpCpuReferenceHead(ref, graph_name.c_str(), /*max_f32=*/16);

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
    if (!g_prefill.Create(qnn.Backend(), ctx.Handle(), profiler.GetProfiler(), "prefill_forward")) {
        std::cerr << "graphCreate for prefill failed\n";
        return -1;
    }

    if (!g_kv.Create(qnn.Backend(), ctx.Handle(), profiler.GetProfiler(), "kv_forward")) {
        std::cerr << "graphCreate for kv failed\n";
        return -1;
    }

    std::cout << "graphCreate OK. graph_handle for prefill=" << g_prefill.Handle() << " for kv= " << g_kv.Handle() << "\n";

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
    if(!RunOneGraph("prefill_forward", qnn.Backend(), g_prefill.Handle(), backendcache, mem, sb, arena, profiler.GetProfiler(), rr_prefill)){
        std::cerr << "Run prefill failed\n";
        return -1;
    }
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

    if(!PostProcessOneGraphRun("prefill_forward", false, rr_prefill.input_ptrs,
            rr_prefill.output_metas, rr_prefill.output_bufs, profiler)){
        return -1;
    }

    if(!PostProcessOneGraphRun("kv_forward", true, rr_kv.input_ptrs,
            rr_kv.output_metas, rr_kv.output_bufs, profiler)){
        return -1;
    }

    sb.ArenaDestroy(arena);

    std::cout << "[QNN] Done.\n";
    return 0;

}