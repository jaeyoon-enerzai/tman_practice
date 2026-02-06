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

static inline void Split(std::vector<std::string>& out,
                         const std::string& s,
                         char delim) {
    out.clear();
    size_t start = 0;
    while (true) {
        size_t pos = s.find(delim, start);
        if (pos == std::string::npos) {
            out.emplace_back(s.substr(start));
            break;
        }
        out.emplace_back(s.substr(start, pos - start));
        start = pos + 1;
    }
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
    if (!CheckQnnOk(api.backendCreate(logger_handle, backend_config, &backend_handle_),
                    "backendCreate")) {
        return false;
    }

    if (!ConfigureOpPackagesFromEnv()) {
        std::cerr << "[QNN] ConfigureOpPackagesFromEnv failed\n";
        return false;
    }

    return true;
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

bool QnnBackendRuntime::ConfigureOpPackagesFromEnv() {
    if (!be_ || !backend_handle_) {
        std::cerr << "[QNN] ConfigureOpPackagesFromEnv: backend not created\n";
        return false;
    }

    // TODO: Expose API to options in QnnManager later (ExecuTorch와 동일)
    std::string opPackagePaths =
        "/data/local/tmp/llama/libQnnTMANOpPackage.so:TMANOpPackageInterfaceProvider:HTP";

    if (const char* env_p = std::getenv("QNN_OP_PACKAGE_PATHS")) {
        opPackagePaths = env_p;  // AOT에서는 "${EXECUTORCH_ROOT}/backends/qualcomm/runtime/op_packages/TMANOpPackage/build/x86_64-linux-clang/libQnnTMANOpPackage.so:TMANOpPackageInterfaceProvider:CPU"
    }

    std::vector<std::string> pkgStrings;
    Split(pkgStrings, opPackagePaths, ',');

    for (const auto& pkgStr : pkgStrings) {
        if (pkgStr.empty()) continue;

        std::vector<std::string> parts;
        Split(parts, pkgStr, ':');

        if (parts.size() != 2 && parts.size() != 3) {
            std::cerr << "[QNN] Malformed opPackageString provided: " << pkgStr << "\n";
            return false;
        }

        const std::string& so_path = parts[0];
        const std::string& provider = parts[1];
        const char* target = (parts.size() == 3) ? parts[2].c_str() : nullptr;

        if (!RegisterOpPackage(so_path, provider, target)) {
            std::cerr << "[QNN] Failed to register op package: " << so_path
                      << " provider=" << provider
                      << " target=" << (target ? target : "(null)") << "\n";
            return false;
        }

        std::cout << "[QNN] Registered Op Package: " << so_path
                  << " provider: " << provider
                  << " target: " << (target ? target : "(null)") << "\n";
    }

    return true;
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