#include "qnn_backendcache.h"

#include <cstdio>
#include <iostream>

#include "QnnCommon.h"  // QNN_GET_ERROR_CODE
#include "HTP/QnnHtpSystemContext.h"  // HTP spill/fill parsing용 (없으면 제거)

static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* what) {
  if (err != QNN_SUCCESS) {
    std::cerr << "[QNN] " << what << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
    return false;
  }
  return true;
}

QnnBackendCacheRuntime::~QnnBackendCacheRuntime() {
  Destroy();
}

bool QnnBackendCacheRuntime::Create(const QnnSystemInterface_t* sys_iface,
                                   const QnnContextBinary& blob) {
  if (!sys_iface) {
    std::cerr << "[QNN] BackendCache Create: sys_iface is null\n";
    return false;
  }
  if (!blob.buffer || blob.nbytes == 0) {
    std::cerr << "[QNN] BackendCache Create: context blob is null/empty\n";
    return false;
  }

  sys_ = sys_iface;
  blob_ = blob;

  if (sys_context_handle_) return true;  // already created

  if (!Configure()) {
    std::cerr << "[QNN] BackendCache Configure failed\n";
    Destroy();
    return false;
  }
  return true;
}

void QnnBackendCacheRuntime::Destroy() {
  if (!sys_ || !sys_context_handle_) return;

  (void)CheckQnnOk(sys_->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sys_context_handle_),
                   "systemContextFree");
  sys_context_handle_ = nullptr;

  state_ = INVALID;
  graph_names_.clear();
  input_tensor_structs_.clear();
  output_tensor_structs_.clear();
}

std::vector<Qnn_Tensor_t> QnnBackendCacheRuntime::GetGraphInputs(
    const std::string& graph_name) const {
  if (state_ != DESERIALIZE) return {};
  auto it = input_tensor_structs_.find(graph_name);
  if (it == input_tensor_structs_.end()) return {};
  return it->second;
}

std::vector<Qnn_Tensor_t> QnnBackendCacheRuntime::GetGraphOutputs(
    const std::string& graph_name) const {
  if (state_ != DESERIALIZE) return {};
  auto it = output_tensor_structs_.find(graph_name);
  if (it == output_tensor_structs_.end()) return {};
  return it->second;
}

bool QnnBackendCacheRuntime::RetrieveBackendBinaryInfo(
    const QnnSystemContext_BinaryInfo_t* /*binaryinfo*/) {
  // 기본은 아무 것도 안 함
  return true;
}

template <typename INFO>
void QnnBackendCacheRuntime::RetrieveGraphInfo(const INFO& info) {
  // graph name
  graph_names_.push_back(info.graphName);

  // inputs
  uint32_t numGraphInputs = info.numGraphInputs;
  auto& in_vec = input_tensor_structs_[graph_names_.back()];
  in_vec.reserve(numGraphInputs);
  for (uint32_t i = 0; i < numGraphInputs; ++i) {
    in_vec.emplace_back(info.graphInputs[i]);
  }

  // outputs
  uint32_t numGraphOutputs = info.numGraphOutputs;
  auto& out_vec = output_tensor_structs_[graph_names_.back()];
  out_vec.reserve(numGraphOutputs);
  for (uint32_t i = 0; i < numGraphOutputs; ++i) {
    out_vec.emplace_back(info.graphOutputs[i]);
  }
}

bool QnnBackendCacheRuntime::GetQnnGraphInfoFromBinary(void* buffer, uint32_t nbytes) {
  if (!sys_ || !sys_context_handle_) {
    std::cerr << "[QNN] GetQnnGraphInfoFromBinary: system context not ready\n";
    return false;
  }

  std::uint32_t num_graphs = 0;
  QnnSystemContext_GraphInfo_t* graphs = nullptr;
  const QnnSystemContext_BinaryInfo_t* binaryinfo = nullptr;
  Qnn_ContextBinarySize_t binaryinfo_size = 0;

  auto err = sys_->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(
      sys_context_handle_, buffer, nbytes, &binaryinfo, &binaryinfo_size);

  if (err != QNN_SUCCESS) {
    std::cerr << "[QNN] qnn_system_context_get_binary_info failed, err="
              << QNN_GET_ERROR_CODE(err) << "\n";
    return false;
  }

  if (!RetrieveBackendBinaryInfo(binaryinfo)) {
    std::cerr << "[QNN] RetrieveBackendBinaryInfo failed\n";
    return false;
  }

  if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    num_graphs = binaryinfo->contextBinaryInfoV1.numGraphs;
    graphs = binaryinfo->contextBinaryInfoV1.graphs;
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    num_graphs = binaryinfo->contextBinaryInfoV2.numGraphs;
    graphs = binaryinfo->contextBinaryInfoV2.graphs;
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    num_graphs = binaryinfo->contextBinaryInfoV3.numGraphs;
    graphs = binaryinfo->contextBinaryInfoV3.graphs;
#endif
  } else {
    std::cerr << "[QNN] Unknown BinaryInfo version " << binaryinfo->version << "\n";
    return false;
  }

  for (std::uint32_t i = 0; i < num_graphs; ++i) {
    if (graphs->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      RetrieveGraphInfo<QnnSystemContext_GraphInfoV1_t>(graphs[i].graphInfoV1);
    } else if (graphs->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
      RetrieveGraphInfo<QnnSystemContext_GraphInfoV2_t>(graphs[i].graphInfoV2);
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
    } else if (graphs->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      RetrieveGraphInfo<QnnSystemContext_GraphInfoV3_t>(graphs[i].graphInfoV3);
#endif
    } else {
      std::cerr << "[QNN] Unknown GraphInfo version " << graphs->version << "\n";
      return false;
    }
  }

  return true;
}

bool QnnBackendCacheRuntime::Configure() {
  // DESERIALIZE only
  state_ = INVALID;
  graph_names_.clear();
  input_tensor_structs_.clear();
  output_tensor_structs_.clear();

  if (!blob_.buffer || blob_.nbytes == 0) {
    std::cerr << "[QNN] Configure: context blob is null/empty\n";
    return false;
  }

  if (!sys_) {
    std::cerr << "[QNN] Configure: sys interface is null\n";
    return false;
  }

  if (!CheckQnnOk(sys_->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sys_context_handle_), "systemContextCreate")) {
    return false;
  }

  state_ = DESERIALIZE;

  if (!GetQnnGraphInfoFromBinary(blob_.buffer, blob_.nbytes)) {
    // we do not consider ONLINE prepare or qcir and multi-graph etc.
    state_ = ONLINE_PREPARE;
    return false;
  }

  if (graph_names_.empty()) {
    std::cerr << "[QNN] Configure: parsed 0 graphs from binary\n";
    return false;
  }

  return true;
}

// ---------------- HTP 파생 ----------------

bool HtpBackendCacheRuntime::RetrieveBackendBinaryInfo(
    const QnnSystemContext_BinaryInfo_t* binaryinfo) {
  if (!binaryinfo) return false;

  QnnHtpSystemContext_HwBlobInfo_t* htp_hwblobinfo = nullptr;
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  QnnHtpSystemContext_GraphBlobInfo_t* htp_graphblobinfo = nullptr;
#endif

  if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV1.hwInfoBlob);
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV2.hwInfoBlob);
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    htp_graphblobinfo = static_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV3.graphs->graphInfoV3.graphBlobInfo);
#endif
  } else {
    return false;
  }

  if (htp_hwblobinfo) {
    if (htp_hwblobinfo->version == QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1) {
      spill_fill_buf_ = (*htp_hwblobinfo).contextBinaryHwInfoBlobV1_t.spillFillBufferSize;
    } else {
      return false;
    }
  }

#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  if (htp_graphblobinfo) {
    if (htp_graphblobinfo->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
      spill_fill_buf_ = (*htp_graphblobinfo).contextBinaryGraphBlobInfoV1.spillFillBufferSize;
    } else {
      return false;
    }
  }
#endif

  return true;
}