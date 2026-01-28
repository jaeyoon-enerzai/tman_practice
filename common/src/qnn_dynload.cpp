#include "qnn_dynload.h"
#include <dlfcn.h>
#include <iostream>

QnnDynLoad& QnnDynLoad::Instance() {
    static QnnDynLoad instance;
    return instance;
}

static void* OpenSo(const std::string& p, int flags){
    void* h = dlopen(p.c_str(), flags);
    if(!h){
        std::cerr << "[QNN] dlopen failed for " << p << ", err=" << dlerror() << "\n";
    }
    std::cout << "dlopen for " << p.c_str() << " succeeded.\n";
    return h; 
}

template<typename Fn>
static Fn LoadSym(void* handle, const char* name){
    // dlsym returns void*, need to cast
    return reinterpret_cast<Fn>(dlsym(handle, name));
}

bool QnnDynLoad::LoadAll(const std::string& backend_so,
                             const std::string& system_so,
                             const QnnSaver_Config_t** saver_config) {
    std::lock_guard<std::mutex> lock(mu_);

    if (qnn_backend_iface_ && qnn_system_iface_) {
        // Already loaded
        return true;
    }
    if (!LoadBackend_(backend_so, saver_config)) {
        return false;
    }
    if (!LoadSystem_(system_so)) {
        return false;
    }
    return true;
}

bool QnnDynLoad::LoadBackend_(const std::string& so_path,
                                   const QnnSaver_Config_t** saver_config) {
    if(qnn_backend_iface_){
        // Already loaded
        return true;
    }

    backend_handle_ = OpenSo(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!backend_handle_) {
        return false;
    }
    // Saver init
    using QnnSaverInitializeFn = decltype(QnnSaver_initialize);
    auto saver_initialize = LoadSym<QnnSaverInitializeFn*>(backend_handle_, "QnnSaver_initialize");
    if (saver_initialize){
        Qnn_ErrorHandle_t err = saver_initialize(saver_config);
        if (err != QNN_SUCCESS) {
            std::cerr << "[QNN] QnnSaver_initialize failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
            return false;
        }
    }

    // Provider
    using QnnInterfaceGetProvidersFn = decltype(QnnInterface_getProviders);
    auto getProviders = LoadSym<QnnInterfaceGetProvidersFn*>(backend_handle_, "QnnInterface_getProviders");
    if (!getProviders) {
        std::cerr << "[QNN] Failed to load symbol QnnInterface_getProviders\n";
        return false;
    }

    std::uint32_t num = 0;
    const QnnInterface_t** provider_list = nullptr;
    Qnn_ErrorHandle_t err = getProviders(&provider_list, &num);
    if (err != QNN_SUCCESS) {
        std::cerr << "[QNN] QnnInterface_getProviders failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
        return false;
    }
    if (num < 1 || provider_list == nullptr || provider_list[0] == nullptr) {
        std::cerr << "[QNN] No providers found in backend\n";
        return false;
    }

    qnn_backend_iface_ = provider_list[0];
    return true;
}

bool QnnDynLoad::LoadSystem_(const std::string& so_path) {
    if(qnn_system_iface_){
        // Already loaded
        return true;
    }

    system_handle_ = OpenSo(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!system_handle_) {
        return false;
    }

    // Provider
    using QnnSystemInterfaceGetProvidersFn = decltype(QnnSystemInterface_getProviders);
    auto getProviders = LoadSym<QnnSystemInterfaceGetProvidersFn*>(system_handle_, "QnnSystemInterface_getProviders");
    if (!getProviders) {
        std::cerr << "[QNN] Failed to load symbol QnnSystemInterface_getProviders\n";
        return false;
    }

    std::uint32_t num = 0;
    const QnnSystemInterface_t** provider_list = nullptr;
    Qnn_ErrorHandle_t err = getProviders(&provider_list, &num);
    if (err != QNN_SUCCESS) {
        std::cerr << "[QNN] QnnSystemInterface_getProviders failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
        return false;
    }
    if (num < 1 || provider_list == nullptr || provider_list[0] == nullptr) {
        std::cerr << "[QNN] No providers found in system\n";
        return false;
    }

    qnn_system_iface_ = provider_list[0];
    return true;
}

bool QnnDynLoad::UnloadAll() {
    std::lock_guard<std::mutex> lock(mu_);

    bool success = true;

    // WARNING : Due to a bug in QNN SDK, unloading and reloading the backend
    // may lead to errors related to memory properties of custom operators.
    // As a workaround, we avoid unloading the backend in normal flow.
    // Only use this function for explicit teardown or testing purposes.
    qnn_backend_iface_ = nullptr;
    qnn_system_iface_ = nullptr;

    if (backend_handle_) {
        if (dlclose(backend_handle_) != 0) {
            std::cerr << "[QNN] dlclose failed for backend, err=" << dlerror() << "\n";
            success = false;
        }
        backend_handle_ = nullptr;
        qnn_backend_iface_ = nullptr;
    }

    if (system_handle_) {
        if (dlclose(system_handle_) != 0) {
            std::cerr << "[QNN] dlclose failed for system, err=" << dlerror() << "\n";
            success = false;
        }
        system_handle_ = nullptr;
        qnn_system_iface_ = nullptr;
    }

    return success;
}