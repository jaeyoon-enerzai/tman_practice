#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <iostream>
#include <vector>

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnGraph.h"
#include "QnnCommon.h"

#include "qnn_graph_config.h"
#include "qnn_profiler.h"
#include "qnn_tensor.h"

#ifndef QNN_OP_VER_PTR
#define QNN_OP_VER_PTR(x) (&((x).v1))
#endif

class QnnGraphRuntime {
public:
  QnnGraphRuntime(): // TODO - config가 아니라 일단 값을 박아넣었음
    htp_graph_cfg_(std::make_unique<QnnHtpGraphCustomConfigRuntime>(
      /*vtcm_mb=*/2,
      /*opt_level=*/3.0f,
      /*enable_dlbc=*/true
    )) {}
  ~QnnGraphRuntime() { Destroy(); }

  QnnGraphRuntime(const QnnGraphRuntime&) = delete;
  QnnGraphRuntime& operator=(const QnnGraphRuntime&) = delete;

  void SetRestoreMode(bool v) { restore_mode_ = v; }

  bool Create(const QnnInterface_t* be_iface,
              Qnn_ContextHandle_t ctx,
              Qnn_ProfileHandle_t profiler,
              const std::string& graph_name);

  void Destroy();

  Qnn_GraphHandle_t Handle() const { return graph_; }
  const std::string& Name() const {return name_;}
  bool IsValid() const { return graph_ != nullptr; }

  bool EnsureTensorInGraph(QnnTensor& t);
  bool AddNode(const Qnn_OpConfig_t& op_config);
  bool Finalize();

private:
  const QnnInterface_t* be_{nullptr};
  Qnn_ContextHandle_t ctx_{nullptr};
  Qnn_GraphHandle_t graph_{nullptr};
  std::string name_;
  std::unique_ptr<QnnHtpGraphCustomConfigRuntime> htp_graph_cfg_;
  std::vector<QnnGraph_Config_t> graph_cfg_storage_;

  Qnn_ProfileHandle_t profiler_{nullptr};
  bool restore_mode_{false};
};