#include "qnn_backend.h"
#include <iostream>
#include <cstdlib>

static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* msg) {
    if (err != QNN_SUCCESS) {
        std::cerr << "[QNN] " << msg << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
        return false;
    }
    return true;
}

QnnBackendRuntime::~QnnBackendRuntime() {
    Destroy();
}

bool QnnBackendRuntime::Create(const QnnInterface_t* be_iface,
                                Qnn_LogHandle_t logger_handle) {
    if (!be_iface) {
        std::cerr << "[QNN] Invalid backend interface\n";
        return false;
    }
    be_ = be_iface;
    if(backend_handle_){
        // Already created
        return true;
    }

    const QnnBackend_Config_t* backend_config[] = {nullptr};

    auto& api = be_->QNN_INTERFACE_VER_NAME;
    return CheckQnnOk(
        api.backendCreate(logger_handle, backend_config, &backend_handle_), "backendCreate");
}

bool QnnBackendRuntime::RegisterOpPackage(const std::string& so_path,
                                        const std::string& interface_provider,
                                        const char* target){
    if (!be_ || !backend_handle_){
        std::cerr << "[QNN] RegisterOpPackage: backend not created\n";
        return false;
    }
    auto &api = be_->QNN_INTERFACE_VER_NAME;
    return CheckQnnOk(api.backendRegisterOpPackage(
        backend_handle_,
        so_path.c_str(),
        interface_provider.c_str(),
        target),
    "backendRegisterOpPackage");
}

void QnnBackendRuntime::Destroy(){
    if(!be_ || !backend_handle_) return;

    auto& api = be_->QNN_INTERFACE_VER_NAME;
    if(!CheckQnnOk(api.backendFree(backend_handle_), "backendFree")){

    }
    backend_handle_ = nullptr;
}

bool QnnBackendRuntime::ValidateOpConfig(const Qnn_OpConfig_t& cfg){
  auto& api = be_->QNN_INTERFACE_VER_NAME;
  return CheckQnnOk(api.backendValidateOpConfig(backend_handle_, cfg),
                    "backendValidateOpConfig");
}