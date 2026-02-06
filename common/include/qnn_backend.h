#pragma once
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnTypes.h"

class QnnBackendRuntime{
    public:
    QnnBackendRuntime() = default;
    ~QnnBackendRuntime();

    QnnBackendRuntime(const QnnBackendRuntime&) = delete;
    QnnBackendRuntime& operator=(const QnnBackendRuntime&) = delete;

    bool Create(const QnnInterface_t* be_iface,
                Qnn_LogHandle_t logger_handle = nullptr);
            
    bool RegisterOpPackage(const std::string& so_path,
                           const std::string& interface_provider,
                           const char* target = nullptr);

    void Destroy();

    Qnn_BackendHandle_t Handle() const { return backend_handle_;}
    bool IsValid() const { return backend_handle_ != nullptr; }
    bool ValidateOpConfig(const Qnn_OpConfig_t& cfg);


    private:
    const QnnInterface_t* be_{nullptr};
    Qnn_BackendHandle_t backend_handle_{nullptr};
    bool ConfigureOpPackagesFromEnv();
};