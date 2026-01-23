#pragma once
#include <cstdint>
#include <mutex>
#include <string>

// QNN headers
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "Saver/QnnSaver.h"

class QnnDynLoad{
    public:
    // keep-alive singleton (workaround for QNN SDK bug described in executorch as below)
    // TODO: [ERROR] [Qnn ExecuTorch]: tcm_migration.cc:174:ERROR:Memory properties specified twice for operator ::TMANLinear
//       The root cause of this error is that when QNN backend is freed, the memory properties of custom ops are not cleared,
//       which will cause the error when the QNN backend is loaded and custom ops are registered again.
//       This is a bug in QNN SDK, related to DEF_TENSOR_PROPERTIES / hnnx::register_tensor_properties.
//       Workaround: prevent the QNN backend from being freed.
    static QnnDynLoad& Instance();

    bool LoadAll(const std::string& backend_so,
                 const std::string& system_so,
                 const QnnSaver_Config_t** saver_config = nullptr);

    const QnnInterface_t* Backend() const { return qnn_backend_iface_;}
    const QnnSystemInterface_t* System() const { return qnn_system_iface_;}

    bool IsBackendLoaded() const { return qnn_backend_iface_ != nullptr; }
    bool IsSystemLoaded() const { return qnn_system_iface_ != nullptr; }

    // For explicit teardown/testing only.
    // In normal flow, DON'T call due to the QNN bug workaround.
    bool UnloadAll();

    private:

    QnnDynLoad() = default;
    ~QnnDynLoad() = default;
    QnnDynLoad(const QnnDynLoad&) = delete;
    QnnDynLoad& operator=(const QnnDynLoad&) = delete;

    bool LoadBackend_(const std::string& so_path,
                      const QnnSaver_Config_t** saver_config);
    bool LoadSystem_(const std::string& so_path);

    mutable std::mutex mu_;

    void* backend_handle_{nullptr};
    void* system_handle_{nullptr};

    const QnnInterface_t* qnn_backend_iface_{nullptr};
    const QnnSystemInterface_t* qnn_system_iface_{nullptr};
};