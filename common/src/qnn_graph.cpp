#include "qnn_graph.h"
#include "QnnCommon.h"
#include "QnnGraph.h"
#include <cstddef>

static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* what) {
  if (err != QNN_SUCCESS) {
    std::cerr << "[QNN] " << what << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
    return false;
  }
  return true;
}

bool QnnGraphRuntime::Create(const QnnInterface_t* be_iface,
                            Qnn_ContextHandle_t ctx,
                            const std::string& graph_name
                          ) {
  if (!be_iface || !ctx) {
    std::cerr << "[QNN] Graph Create: be_iface or ctx is null\n";
    return false;
  }
  // Not taken multi-graph into account
  if (graph_) return true;

  be_ = be_iface;
  ctx_ = ctx;
  name_ = graph_name;

  auto& api = be_->QNN_INTERFACE_VER_NAME;

  Qnn_ErrorHandle_t err = QNN_SUCCESS;

  if(restore_mode_){
    err = api.graphRetrieve(ctx_, name_.c_str(), &graph_);

    // TODO - profiler (see QnnGraphCommon.cpp - L69)
    return CheckQnnOk(err, "graphRetrieve");
  }

  std::vector<const QnnGraph_Config_t*> cfg_ptrs;
  cfg_ptrs.clear();
  graph_cfg_storage_.clear();

  const auto custom = htp_graph_cfg_->Create();

  graph_cfg_storage_.resize(custom.size());
  cfg_ptrs.reserve(custom.size() + 1);
  for(size_t i=0; i< custom.size(); ++i){
    graph_cfg_storage_[i].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_cfg_storage_[i].customConfig = custom[i];
    cfg_ptrs.push_back(&graph_cfg_storage_[i]);
  }
  cfg_ptrs.push_back(nullptr);

  const QnnGraph_Config_t* const* cfg_cptr =
      cfg_ptrs.empty() ? nullptr : cfg_ptrs.data();
  const QnnGraph_Config_t** cfg_pp =
      cfg_cptr ? const_cast<const QnnGraph_Config_t**>(cfg_cptr) : nullptr;

  // Graph-Compilation - graph retrieve는 나중 TODO
  err = api.graphCreate(
      ctx_,
      name_.c_str(),
      cfg_pp,
      &graph_);

  return CheckQnnOk(err, "graphCreate");
}

void QnnGraphRuntime::Destroy() {
  // QNN은 보통 graphFree API가 별도로 없고, contextFree 때 같이 정리되는 경우가 많음.
  // (SDK에 graphFree가 있으면 여기서 호출하면 됨)
  graph_ = nullptr;
  ctx_ = nullptr;
  be_ = nullptr;
}

bool QnnGraphRuntime::EnsureTensorInGraph(QnnTensor& t){
  if(!be_ || !graph_) return false;
  if (t.IsCreated()) return true;
  
  Qnn_Tensor_t tensor = t.Clone();
  auto& api = be_->QNN_INTERFACE_VER_NAME;

  Qnn_ErrorHandle_t err = api.tensorCreateGraphTensor(graph_, &tensor);

  int name_conflict = 0;
  while(err == QNN_TENSOR_ERROR_NAME_HASH_COLLISION){
      const std::string old = t.Name();
      std::string nn = old + "_" + std::to_string(name_conflict++);
      std::cout << "Name collision. Change names from " << old.c_str() << " to " << nn.c_str() << std::endl;
      t.SetName(nn);
      QNN_TENSOR_VER_PTR(tensor)->name = t.Name().c_str();
      err = api.tensorCreateGraphTensor(graph_, &tensor);
  }
  if (err != QNN_SUCCESS){
    std::cerr << "[QNN] tensorCreateGraphTensor failed, err= " << QNN_GET_ERROR_CODE(err) << " name = " << t.Name() << "\n";
    return false;
  }
  std::cout << "Tensor with name [ " << t.Name() << " ] is registered" << std::endl;

  t.UpdateMetaFrom(tensor);
  t.MarkCreated();
  return true;
}

bool QnnGraphRuntime::AddNode(const Qnn_OpConfig_t& op_config) {
  if (!be_ || !graph_) return false;
  auto& api = be_->QNN_INTERFACE_VER_NAME;
  return CheckQnnOk(api.graphAddNode(graph_, op_config), "graphAddNode");
}

bool QnnGraphRuntime::Finalize() {
  if (!be_ || !graph_) return false;
  auto& api = be_->QNN_INTERFACE_VER_NAME;
  // profileHandle/signalHandle 지금은 nullptr
  return CheckQnnOk(api.graphFinalize(graph_, /*profile=*/nullptr, /*signal=*/nullptr), "graphFinalize");
}