#include <iostream>
#include <cstdint>
#include <random>
#include <sys/types.h>
#include <vector>
#include <cstddef>
#include <string>
#include <fstream>
#include "QnnLog.h"
#include "QnnTypes.h"
#include "qnn_device.h"
#include "qnn_dynload.h"
#include "qnn_backend.h"
#include "qnn_context.h"
#include "qnn_graph.h"
#include "qnn_tensor.h"
#include "qnn_profiler.h"
#include "qnn_log.h"

#define GROUP_SIZE 128
#define SYMMETRIC 1
#define BITS 4

int _get_c_size(int m, int bits){
  int c_size = m * bits;
  return c_size * 4;
}

int _get_l_size(int k, int group_size, bool need_dequant){
  int LUT_G = 4;
  int LUT_SIZE = 16;
  int ACT_GROUP_SIZE = 256;
  // float16
  int x_size = need_dequant? k : 0;
  // int16
  int l_size = k / LUT_G * LUT_SIZE;
  // float32
  int ls_size = (ACT_GROUP_SIZE == -1) ? 1 : k / ACT_GROUP_SIZE;
  // float32
  int lb_size = (group_size == 0) ? 1 : k / group_size;
  int eff_ls_size = (ls_size * 4 > 128) ? (ls_size*4) : 128;
  int eff_lb_size = (lb_size * 4 / 128) ? (lb_size*4) : 128; 
  return x_size * 2 + l_size * 2 + eff_ls_size + eff_lb_size;
}

static bool save_f32_raw(const std::string& path, const float* data, size_t numel) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Failed to open for write: " << path << "\n";
    return false;
  }
  out.write(reinterpret_cast<const char*>(data), sizeof(float) * numel);
  if (!out.good()) {
    std::cerr << "Write failed: " << path << "\n";
    return false;
  }
  return true;
}

struct OpHolder {
  std::string name_store;
  std::vector<Qnn_Tensor_t> inputs;
  std::vector<Qnn_Tensor_t> outputs;
  std::vector<Qnn_Param_t> params;

  Qnn_OpConfig_t cfg;  // 여기서는 그냥 선언만!

  OpHolder() : cfg(QNN_OPCONFIG_INIT) {}   // (1) 생성자에서 "대입 초기화"로 처리

  void bind(const char* package, const char* type) {
    cfg = QNN_OPCONFIG_INIT;               // (2) 필요하면 여기서도 리셋
    cfg.version = QNN_OPCONFIG_VERSION_1;
    auto* c = QNN_OP_VER_PTR(cfg);

    c->name = name_store.c_str();
    c->packageName = package;
    c->typeName = type;

    c->numOfParams = static_cast<uint32_t>(params.size());
    c->params = params.empty() ? nullptr : params.data();

    c->numOfInputs = static_cast<uint32_t>(inputs.size());
    c->inputTensors = inputs.empty() ? nullptr : inputs.data();

    c->numOfOutputs = static_cast<uint32_t>(outputs.size());
    c->outputTensors = outputs.empty() ? nullptr : outputs.data();
  }

  // tensor param은 TODO
  void addScalarU32(const char* name, uint32_t v) {
    Qnn_Param_t p{};
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name      = name;
    p.scalarParam.dataType = QNN_DATATYPE_UINT_32;
    p.scalarParam.uint32Value = v;
    params.push_back(p);
  }
  void addScalarI32(const char* name, int v) {
    Qnn_Param_t p{};
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name      = name;
    p.scalarParam.dataType = QNN_DATATYPE_INT_32;
    p.scalarParam.int32Value = v;
    params.push_back(p);
  }
  void addScalarB8(const char*name, uint8_t v){
    Qnn_Param_t p{};
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name      = name;
    p.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    p.scalarParam.uint8Value = v;
    params.push_back(p);
  }
};


template <typename F>
static OpHolder MakeOpHolder(const std::string& name,
                                const char* package,
                                const char* type,
                                const QnnTensor& x,
                                const QnnTensor* y,
                                const QnnTensor* z,
                                const QnnTensor& out,
                                F fillParams // callback
                            ) {
  OpHolder h;
  h.name_store = name;
  if (y == nullptr){
    h.inputs = {x.Clone()};
  } else if (z == nullptr){
    h.inputs = { x.Clone(), y->Clone() };
  } else{
    h.inputs = { x.Clone(), y->Clone(), z->Clone()};
  }
  h.outputs = { out.Clone() };
  h.params.clear();

  // 디버그 출력(원하면 유지)
  std::cout << "[MakeAddOpHolder] " << name << "\n";
  std::cout << "  input0 id: " << QNN_TENSOR_VER_PTR(h.inputs[0])->id
            << " size: " << QNN_TENSOR_VER_PTR(h.inputs[0])->clientBuf.dataSize << "\n";
  std::cout << "  output id: " << QNN_TENSOR_VER_PTR(h.outputs[0])->id
            << " size: " << QNN_TENSOR_VER_PTR(h.outputs[0])->clientBuf.dataSize << "\n";

  fillParams(h);
  h.bind(package, type);
  return h;  // 값으로 반환해도 내부 문자열/vector는 이동(move)되어 안전
}


static bool BuildOneGraph(
    QnnBackendRuntime& backend,
    QnnGraphRuntime& graph,
    bool is_kv,
    // (필요하면) seed나 차이 주는 파라미터 추가 가능
    unsigned int B, unsigned int L, unsigned int D, unsigned int C,
    uint8_t* static_v, uint8_t* static_sc, float* static_q, float* static_k,
    unsigned int v_bytes, unsigned int qk_bytes, unsigned int scale_bytes
) {
  // QBIT PARAM
  bool ADD_CONVERT = true;

  // ---- Tensor 정의 ----
  std::vector<uint32_t> x_dims{B, L, C};
  std::vector<uint32_t> y_dims{C, C};
  std::vector<uint32_t> v_dims{C, D};
  std::vector<uint32_t> weight_dims{D, C}; // outch, inch
  std::vector<uint32_t> flatten_o_dims{B * L, D};
  std::vector<uint32_t> o_dims{B, L, D};
  std::vector<uint32_t> attn_dims{B, L, L};
  
  std::vector<uint32_t> flat_x_dims{B*L, C};
  std::vector<uint32_t> l_tns_dims{1, static_cast<uint32_t>(_get_l_size(C, GROUP_SIZE, !ADD_CONVERT))};
  std::vector<uint32_t> v_qbit_dims{1, D * C / 2};
  std::vector<uint32_t> scale_dims{1, D * C / BITS / GROUP_SIZE * 4};
  std::vector<uint32_t> c_tns_dims{1, static_cast<uint32_t>(_get_c_size(D, BITS))};

  // ⚠️ 중요:
  // 같은 weight sharing을 노리면 wq/wk/wvprime 같은 STATIC 텐서는
  // 두 graph에서 "이름이 동일"해야 할 가능성이 매우 큼.
  // (지금은 일단 동일 name 유지)
  QnnTensor x("x",   QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, x_dims);
  QnnTensor y("y",   QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, y_dims);

  QnnTensor wq("wq", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32, weight_dims,
               nullptr, qk_bytes, static_cast<const void*>(static_q));
  QnnTensor wk("wk", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32, weight_dims,
               nullptr, qk_bytes, static_cast<const void*>(static_k));
  QnnTensor wvprime("wvprime", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_8, v_qbit_dims,
                    nullptr, D * C, static_cast<const void*>(static_v));
  std::unique_ptr<QnnTensor> wv_ptr, cast_x_ptr, l_tns_ptr, scale_ptr, c_tns_ptr, vflat_ptr, cast_v_ptr, flat_x_ptr;
  if(!is_kv){
    wv_ptr = std::make_unique<QnnTensor>(
        "wv", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, v_dims
    );
  } else{
    if(B != 1 || L != 1){
      std::cerr << "Decoding does not support batch and sequence length more than 1" << std::endl;
    }
    flat_x_ptr = std::make_unique<QnnTensor>("flat_x_ptr", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, flat_x_dims);
    cast_x_ptr = std::make_unique<QnnTensor>(
      "cast_x_tns", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, flat_x_dims 
    );
    l_tns_ptr = std::make_unique<QnnTensor>("l_tns", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UINT_8, l_tns_dims);
    scale_ptr = std::make_unique<QnnTensor>("scale", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, scale_dims, nullptr, scale_bytes, static_cast<const void*>(static_sc));
    c_tns_ptr = std::make_unique<QnnTensor>("c_tns", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UINT_8, c_tns_dims);
    vflat_ptr = std::make_unique<QnnTensor>("vflat_tns", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, flatten_o_dims);
    cast_v_ptr = std::make_unique<QnnTensor>("cast_v_tns", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, o_dims);
  }

  QnnTensor q("q", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, flatten_o_dims);
  QnnTensor qprime("qprime", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, o_dims);
  QnnTensor k("k", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, flatten_o_dims);
  QnnTensor kprime("kprime", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, o_dims);
  QnnTensor v("v", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, o_dims);
  QnnTensor attn("attn", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, attn_dims);
  QnnTensor out("o", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, o_dims);

  // ---- Graph tensor 등록 ----
  if (!graph.EnsureTensorInGraph(x)) return false;
  if (!graph.EnsureTensorInGraph(y)) return false;
  if (!graph.EnsureTensorInGraph(wq)) return false;
  if (!graph.EnsureTensorInGraph(wk)) return false;
  if (!graph.EnsureTensorInGraph(wvprime)) return false;
  if(!is_kv){
    if (!graph.EnsureTensorInGraph(*wv_ptr)) return false;
  } else{
    if (!graph.EnsureTensorInGraph(*flat_x_ptr)) return false;
    if (!graph.EnsureTensorInGraph(*cast_x_ptr)) return false;
    if (!graph.EnsureTensorInGraph(*l_tns_ptr)) return false;
    if (!graph.EnsureTensorInGraph(*scale_ptr)) return false;
    if (!graph.EnsureTensorInGraph(*c_tns_ptr)) return false;
    if (!graph.EnsureTensorInGraph(*vflat_ptr)) return false;
    if (!graph.EnsureTensorInGraph(*cast_v_ptr)) return false;

  }
  if (!graph.EnsureTensorInGraph(q)) return false;
  if (!graph.EnsureTensorInGraph(qprime)) return false;
  if (!graph.EnsureTensorInGraph(k)) return false;
  if (!graph.EnsureTensorInGraph(kprime)) return false;
  if (!graph.EnsureTensorInGraph(v)) return false;
  if (!graph.EnsureTensorInGraph(attn)) return false;
  if (!graph.EnsureTensorInGraph(out)) return false;

  const char* kPackage = "qti.aisw";

  // ---- Op 만들기 ----

  // q = x * wq, k = x * wk
  // prefill :  wv = wvprime * y, v = x * wv
  // decoding : v = x * wvprime -> tmanlinear(x, wvprime)
  //     => cast_x = convert(x), l = precompute(x, scale), c = tmanlinear(l, w), out = finalize(c), cast_out = cast(out)

  OpHolder matmul_q = MakeOpHolder("matmul_q", kPackage, "FullyConnected", x, &wq, nullptr, q,
                                   [&](OpHolder& oh){ oh.addScalarB8("keep_dims", 0); });
  OpHolder matmul_k = MakeOpHolder("matmul_k", kPackage, "FullyConnected", x, &wk, nullptr, k,
                                   [&](OpHolder&){});
  OpHolder matmul_wv, matmul_v, reshape_x, cast_x, precompute, tmanlinear, finalize, reshape_v, cast_v;
  if(!is_kv){
    std::cout << "Temporaily removed value linear" << std::endl;
    // matmul_wv  = MakeOpHolder("matmul_wv", kPackage, "MatMul", wvprime, &y, nullptr, *wv_ptr,
    //                                  [&](OpHolder&){});
    // matmul_v = MakeOpHolder("matmul_v", kPackage, "MatMul", x, wv_ptr.get(), nullptr, v,
    //                                [&](OpHolder& oh){ oh.addScalarB8("transpose_in1", 1); });
    matmul_v = MakeOpHolder("matmul_v", kPackage, "MatMul", x, &wq, nullptr, v, [&](OpHolder& oh){});
  } else{
    reshape_x = MakeOpHolder("reshape_x", kPackage, "Reshape", x, nullptr, nullptr, *flat_x_ptr, [&](OpHolder&){});
    cast_x = MakeOpHolder("cast_x", kPackage, "Cast", *flat_x_ptr.get(), nullptr, nullptr, *cast_x_ptr, [&](OpHolder&){});
    precompute = MakeOpHolder("precompute", "TMANOpPackage", "TMANPrecompute", *cast_x_ptr.get(), nullptr, nullptr, *l_tns_ptr, [&](OpHolder& oh){
      oh.addScalarI32("group_size", GROUP_SIZE);
      oh.addScalarI32("bits", BITS);
      oh.addScalarI32("symmetric", SYMMETRIC);
    });

    tmanlinear = MakeOpHolder("tmanlinear", "TMANOpPackage", "TMANLinear", *l_tns_ptr.get(), &wvprime, scale_ptr.get(), *c_tns_ptr.get(), [&](OpHolder& oh){
      oh.addScalarI32("group_size", GROUP_SIZE);
      oh.addScalarI32("bits", BITS);
      oh.addScalarI32("symmetric", SYMMETRIC);
    });
    finalize = MakeOpHolder("finalize", "TMANOpPackage", "TMANFinalize", *c_tns_ptr.get(), nullptr, nullptr, *vflat_ptr.get(), [&](OpHolder& oh){
      oh.addScalarI32("group_size", GROUP_SIZE);
      oh.addScalarI32("bits", BITS);
      oh.addScalarI32("symmetric", SYMMETRIC);
    });
    
    reshape_v = MakeOpHolder("reshape_v", kPackage, "Reshape", *vflat_ptr.get(), nullptr, nullptr, *cast_v_ptr.get(), [&](OpHolder&){});
    cast_v = MakeOpHolder("cast_v", kPackage, "Cast", *cast_v_ptr.get(), nullptr, nullptr, v, [&](OpHolder&){});
  }
  OpHolder reshape_q = MakeOpHolder("reshape_q", kPackage, "Reshape", q, nullptr, nullptr, qprime,
                                    [&](OpHolder&){});
  OpHolder reshape_k = MakeOpHolder("reshape_k", kPackage, "Reshape", k, nullptr, nullptr, kprime,
                                    [&](OpHolder&){});
  OpHolder matmul_attn = MakeOpHolder("matmul_attn", kPackage, "MatMul", qprime, &kprime, nullptr, attn,
                                      [&](OpHolder& oh){ oh.addScalarB8("transpose_in1", 1); });
  OpHolder matmul_o = MakeOpHolder("matmul_o", kPackage, "MatMul", attn, &v, nullptr, out,
                                   [&](OpHolder&){});

  // ---- Validate + AddNode ----
  auto validate_and_add = [&](OpHolder& op, const char* tag) -> bool {
    std::cout << "Validate and add : " << tag << std::endl;
    if (!backend.ValidateOpConfig(op.cfg)) {
      std::cout << "c shape : (1, " << c_tns_dims[1] << ") " << std::endl;
      std::cerr << "ValidateOpConfig failed: " << tag << "\n";
      return false;
    }
    std::cout << "Validated Op " << tag << " and now adding\n";
    if (!graph.AddNode(op.cfg)) {
      std::cerr << "AddNode failed: " << tag << "\n";
      return false;
    }
    std::cout << "Added Node " << tag << "\n";
    return true;
  };

  if (!validate_and_add(matmul_q, "matmul_q")) return false;
  if (!validate_and_add(matmul_k, "matmul_k")) return false;
  if(!is_kv){
    // if (!validate_and_add(matmul_wv, "matmul_wv")) return false;
    if (!validate_and_add(matmul_v, "matmul_v")) return false;
  } else{
    if (!validate_and_add(reshape_x, "reshape_x")) return false;
    if (!validate_and_add(cast_x, "cast_x")) return false;
    if (!validate_and_add(precompute, "precompute")) return false;
    if (!validate_and_add(tmanlinear, "tmanlinear")) return false;
    if (!validate_and_add(finalize, "finalize")) return false;
    if (!validate_and_add(reshape_v, "reshape_v")) return false;
    if (!validate_and_add(cast_v, "cast_v")) return false;
  }
  if (!validate_and_add(reshape_q, "reshape_q")) return false;
  if (!validate_and_add(reshape_k, "reshape_k")) return false;
  if (!validate_and_add(matmul_attn, "matmul_attn")) return false;
  // if (!validate_and_add(matmul_v, "matmul_v")) return false;
  if (!validate_and_add(matmul_o, "matmul_o")) return false;

  // ---- Finalize ----
  if (!graph.Finalize()) return false;

  return true;
}

template <typename T>
bool load_raw(const std::string& path, std::vector<T>& out, size_t numel) {
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

  // 파일이 더 큰 경우는 허용(원하면 체크 가능)
  return true;
}

int main(int argc, char** argv) {
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
    if (!CreateQnnLogger(qnn.Backend(), &logHandle, QNN_LOG_LEVEL_VERBOSE)) {
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
    
    // custom op package registesr
    // backend.RegisterOpPackage("libQnnTMANOpPackage.so", "TMANOpPackageInterfaceProvider", "HTP");

    QnnDeviceRuntime device;
    if(!device.Create(qnn.Backend(), /*logger_handler=*/logHandle)){
        std::cerr << "deviceCreate failed\n";
        return -1;
    }
    std::cout << "deviceCreate OK\n";

    QnnProfilerRuntime profiler;
    if(!profiler.Create(qnn.Backend(), qnn.System(), backend.Handle(), QnnProfileLevel::Optrace, true, "qnn.log")){
        std::cerr << "ProfilerCreate failed\n";
        return -1;
    }

    QnnContextRuntime ctx;
    ctx.SetWeightSharing(true);
    if(!ctx.Create(qnn.Backend(), backend.Handle(), device.Handle())){
        std::cerr << "contextCreate failed\n";
        return -1;
    }

    std::cout << "contextCreate OK\n";

    QnnGraphRuntime graph_kv, graph_prefill;
    graph_kv.SetRestoreMode(false);
    graph_prefill.SetRestoreMode(false);
    if (!graph_kv.Create(qnn.Backend(), ctx.Handle(), profiler.GetProfiler(), "kv_forward")) {
        std::cerr << "graphCreate for kv graph failed\n";
        return -1;
    }
    // if (!graph_prefill.Create(qnn.Backend(), ctx.Handle(), profiler.GetProfiler(), "prefill_forward")) {
    //     std::cerr << "graphCreate for prefill graph failed\n";
    //     return -1;
    // }

    std::cout << "graphCreate OK. kv graph_handle=" << graph_kv.Handle() << " and prefill graph handle=" << graph_prefill.Handle() << "\n";

    // randomize static tensor data
    unsigned int B = 1;
    unsigned int L = 1; // 일단
    unsigned int D = 2048;
    unsigned int C = 2048;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    unsigned int v_bytes = static_cast<uint32_t>(D * C * sizeof(uint8_t) / 2);
    // float* static_v = new float[D*C];
    // for (size_t i=0; i< D*C; i++) static_v[i] = dist(rng);
    unsigned int qk_bytes = static_cast<uint32_t>(D * C * sizeof(float));
    unsigned int scale_bytes = static_cast<uint32_t>(D * C  / BITS/ GROUP_SIZE * 4 * sizeof(uint16_t));
    float* static_q = new float[D*C];
    for (size_t i=0; i< D*C; i++) static_q[i] = dist(rng);
    float* static_k = new float[D*C];
    for (size_t i=0; i< D*C; i++) static_k[i] = dist(rng);

    // save static tensor
    // save_f32_raw("static_v.bin",static_cast<const float*>(static_v), D*C);
    save_f32_raw("static_q.bin", static_cast<const float*>(static_q), D*C);
    save_f32_raw("static_k.bin", static_cast<const float*>(static_k), D*C);

    std::vector<uint8_t> static_v;
    if(!load_raw("/workspace/m2048_k8192_g128/w_repacked.bin", static_v, D*C/2)) return -1;
    std::vector<uint8_t> static_sc;
    if(!load_raw("/workspace/m2048_k8192_g128/s_repacked.bin", static_sc, D*C/BITS/GROUP_SIZE*4*2)) return -1;

    // if(!BuildOneGraph(backend, graph_prefill, false, B, L, D, C, nullptr, nullptr, static_q, static_k, v_bytes, qk_bytes)){
    //     std::cerr << "BuildOneGraph for prefill graph failed\n";
    //     return -1;
    // }
    // std::cout << "Build Prefill Graph\n";

    if(!BuildOneGraph(backend, graph_kv, true, B, L, D, C, static_v.data(), static_sc.data(), static_q, static_k, v_bytes, qk_bytes, scale_bytes)){
        std::cerr << "BuildOneGraph for kv graph failed\n";
        return -1;
    }
    std::cout << "Build KV Graph\n";
    
    std::vector<uint8_t> blob;
    if (!ctx.GetBinary(blob)) return -1;

    std::ofstream ofs("multi_graph.bin", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(blob.data()), blob.size());
    ofs.close();

    std::cout << "OK: wrote context binary multi_graph.bin (" << blob.size() << " bytes)\n";

    delete[] static_q;
    delete[] static_k;

    // scope 종료 시 backend Destroy
    return 0;
    

}