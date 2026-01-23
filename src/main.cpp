#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include "qnn_device.h"
#include "qnn_dynload.h"
#include "qnn_backend.h"
#include "qnn_context.h"
#include "qnn_graph.h"
#include "qnn_tensor.h"

static Qnn_OpConfig_t MakeAddOp(const std::string& name,
                               const char* package,
                               const char* type,
                               const QnnTensor& x,
                               const QnnTensor& y,
                               const QnnTensor& out) {
  
  static std::string name_store;
  static std::vector<Qnn_Tensor_t> inputs;
  static std::vector<Qnn_Tensor_t> outputs;
  static std::vector<Qnn_Param_t> params;

  name_store = name;
  inputs = { x.Clone(), y.Clone() };

  std::cout << "input0 id : " << QNN_TENSOR_VER_PTR(inputs[0])->id << " size : " << QNN_TENSOR_VER_PTR(inputs[0])->clientBuf.dataSize << std::endl;
  std::cout << "input1 id : " << QNN_TENSOR_VER_PTR(inputs[1])->id << " size : " << QNN_TENSOR_VER_PTR(inputs[1])->clientBuf.dataSize << std::endl;
  outputs = { out.Clone() };
  std::cout << "output id : " << QNN_TENSOR_VER_PTR(outputs[0])->id << " size : " << QNN_TENSOR_VER_PTR(outputs[0])->clientBuf.dataSize << std::endl;
  params.clear(); // Add는 보통 param 없음

  Qnn_OpConfig_t cfg = QNN_OPCONFIG_INIT;
  cfg.version = QNN_OPCONFIG_VERSION_1;
  auto* c = QNN_OP_VER_PTR(cfg);

  c->name = name_store.c_str();
  c->packageName = package;
  c->typeName = type;
  c->numOfParams = 0;
  c->params = nullptr;
  c->numOfInputs = (uint32_t)inputs.size();
  c->inputTensors = inputs.data();
  c->numOfOutputs = (uint32_t)outputs.size();
  c->outputTensors = outputs.data();
  return cfg;
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
    
    QnnBackendRuntime backend;
    if (!backend.Create(qnn.Backend(), /*logger_handler=*/nullptr)){
        std::cerr << "backendCreate failed\n";
        return -1;
    }
    std::cout << "backendCreate OK\n";
    
    // custom op package registesr
    // backend.RegisterOpPackage("libQnnTMANOpPackage.so", "TMANOpPackageInterfaceProvider", "HTP");

    QnnDeviceRuntime device;
    if(!device.Create(qnn.Backend(), /*logger_handler=*/nullptr)){
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
    std::vector<uint32_t> dims{4};
    QnnTensor x("x",   QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, dims);
    QnnTensor y("y",   QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, dims);
    QnnTensor out("o", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, dims);

std::cout
        << "x id=" << QNN_TENSOR_VER_PTR(*x.Tensor())->id
        << " y id=" << QNN_TENSOR_VER_PTR(*y.Tensor())->id
        << " o id=" << QNN_TENSOR_VER_PTR(*out.Tensor())->id
        << "\n";


    if (!graph.EnsureTensorInGraph(x)) return -1;
    if (!graph.EnsureTensorInGraph(y)) return -1;
    if (!graph.EnsureTensorInGraph(out)) return -1;

    // executorch가 쓰는 값으로 맞춰라 (아래는 흔한 예시 문자열)
    const char* kPackage = "qti.aisw";
    const char* kType    = "ElementWiseAdd";
    std::string add_name = "add0";

    Qnn_OpConfig_t add = MakeAddOp(add_name, kPackage, kType, x, y, out);

    std::cout
        << "x id=" << QNN_TENSOR_VER_PTR(*x.Tensor())->id
        << " y id=" << QNN_TENSOR_VER_PTR(*y.Tensor())->id
        << " o id=" << QNN_TENSOR_VER_PTR(*out.Tensor())->id
        << "\n";

    if(!backend.ValidateOpConfig(add)) return -1;

    if (!graph.AddNode(add)) return -1;
    std::cout << "Added Op Node\n";
    // if (!graph.Finalize()) return -1;

    // std::vector<uint8_t> blob;
    // if (!ctx.GetBinary(blob)) return -1;

    // std::ofstream ofs("add_graph.bin", std::ios::binary);
    // ofs.write(reinterpret_cast<const char*>(blob.data()), blob.size());
    // ofs.close();

    // std::cout << "OK: wrote context binary add_graph.bin (" << blob.size() << " bytes)\n";
    // scope 종료 시 backend Destroy
    return 0;
    

}