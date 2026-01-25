#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include "QnnLog.h"
#include "QnnTypes.h"
#include "qnn_device.h"
#include "qnn_dynload.h"
#include "qnn_backend.h"
#include "qnn_context.h"
#include "qnn_graph.h"
#include "qnn_tensor.h"
#include "qnn_log.h"

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
};

static OpHolder MakeAddOpHolder(const std::string& name,
                                const char* package,
                                const char* type,
                                const QnnTensor& x,
                                const QnnTensor& y,
                                const QnnTensor& out) {
  OpHolder h;
  h.name_store = name;
  h.inputs = { x.Clone(), y.Clone() };
  h.outputs = { out.Clone() };
  h.params.clear();

  // 디버그 출력(원하면 유지)
  std::cout << "[MakeAddOpHolder] " << name << "\n";
  std::cout << "  input0 id: " << QNN_TENSOR_VER_PTR(h.inputs[0])->id
            << " size: " << QNN_TENSOR_VER_PTR(h.inputs[0])->clientBuf.dataSize << "\n";
  std::cout << "  input1 id: " << QNN_TENSOR_VER_PTR(h.inputs[1])->id
            << " size: " << QNN_TENSOR_VER_PTR(h.inputs[1])->clientBuf.dataSize << "\n";
  std::cout << "  output id: " << QNN_TENSOR_VER_PTR(h.outputs[0])->id
            << " size: " << QNN_TENSOR_VER_PTR(h.outputs[0])->clientBuf.dataSize << "\n";

  h.bind(package, type);
  return h;  // 값으로 반환해도 내부 문자열/vector는 이동(move)되어 안전
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

    QnnContextRuntime ctx;
    if(!ctx.Create(qnn.Backend(), backend.Handle(), device.Handle(), /*cfg=*/nullptr)){
        std::cerr << "contextCreate failed\n";
        return -1;
    }

    std::cout << "contextCreate OK\n";

    QnnGraphRuntime graph;
    if (!graph.Create(qnn.Backend(), ctx.Handle(), "empty_graph", /*cfg=*/nullptr)) {
        std::cerr << "graphCreate failed\n";
        return -1;
    }

    std::cout << "graphCreate OK. graph_handle=" << graph.Handle() << "\n";

    // x,y,out 텐서 1D [4] float32 예시
    std::vector<uint32_t> dims{1,8};
    QnnTensor x("x",   QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, dims);
    QnnTensor y("y",   QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, dims);
    QnnTensor z("z",   QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, dims);
    QnnTensor out("o", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, dims);

std::cout
        << "x id=" << QNN_TENSOR_VER_PTR(*x.Tensor())->id
        << " y id=" << QNN_TENSOR_VER_PTR(*y.Tensor())->id
        << " o id=" << QNN_TENSOR_VER_PTR(*out.Tensor())->id
        << "\n";


    if (!graph.EnsureTensorInGraph(x)) return -1;
    if (!graph.EnsureTensorInGraph(y)) return -1;
    if (!graph.EnsureTensorInGraph(z)) return -1;
    if (!graph.EnsureTensorInGraph(out)) return -1;

    // executorch가 쓰는 값으로 맞춰라 (아래는 흔한 예시 문자열)
    const char* kPackage = "qti.aisw";
    const char* kType    = "ElementWiseAdd";
    std::string add_name = "add0";

    OpHolder add0 = MakeAddOpHolder(add_name, kPackage, kType, x, y, z);
    OpHolder add1 = MakeAddOpHolder("add1", kPackage, kType, x, z, out);

    std::cout
        << "x id=" << QNN_TENSOR_VER_PTR(*x.Tensor())->id
        << " y id=" << QNN_TENSOR_VER_PTR(*y.Tensor())->id
        << " o id=" << QNN_TENSOR_VER_PTR(*out.Tensor())->id
        << "\n";

    if(!backend.ValidateOpConfig(add0.cfg)){
        std::cout << "Something is wrong in OpConfig\n";
        return -1;
    }

    std::cout << "VALIDATED add0\n";

    if (!graph.AddNode(add0.cfg)) return -1;
    std::cout << "Added add0 Op Node\n";

    if(!backend.ValidateOpConfig(add1.cfg)){
        std::cout << "Something is wrong in OpConfig\n";
        return -1;
    }

    std::cout << "VALIDATED add1\n";

    if (!graph.AddNode(add1.cfg)) return -1;
    std::cout << "Added add1 Op Node\n";

    if (!graph.Finalize()) return -1;

    std::vector<uint8_t> blob;
    if (!ctx.GetBinary(blob)) return -1;

    std::ofstream ofs("add_graph.bin", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(blob.data()), blob.size());
    ofs.close();

    std::cout << "OK: wrote context binary add_graph.bin (" << blob.size() << " bytes)\n";
    // scope 종료 시 backend Destroy
    return 0;
    

}