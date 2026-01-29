#pragma once
#include <cstdint>
#include <iostream>
#include <memory>
#include <cstring>
#include <vector>

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnContext.h"

#include "HTP/QnnHtpContext.h"

class QnnContextRuntime{
    public:
    QnnContextRuntime() = default;
    ~QnnContextRuntime() { Destroy();}

    QnnContextRuntime(const QnnContextRuntime&) = delete;
    QnnContextRuntime& operator=(const QnnContextRuntime&) = delete;

    bool Create(const QnnInterface_t* be,
                Qnn_BackendHandle_t backend_handle,
                Qnn_DeviceHandle_t device_handle_);

    bool CreateFromBinary(const QnnInterface_t* be,
                Qnn_BackendHandle_t backend_handle,
                Qnn_DeviceHandle_t device_handle,
                Qnn_ProfileHandle_t profileHandle,
            const uint8_t* ctx_bin,
            uint32_t ctx_bin_bytes);

    bool GetBinary(std::vector<uint8_t>& out_blob);

    void Destroy();

    Qnn_ContextHandle_t Handle() const {return ctx_;}
    bool IsValid() const { return ctx_ != nullptr;}

    void SetWeightSharing(bool on) {weight_sharing_ = on;}
    void SetMultiContext(bool on, uint64_t max_sf_buf_size){
        use_multi_contexts_ = on;
        max_sf_buf_size_ = max_sf_buf_size;
    }

    private:
    bool MakeConfig(std::vector<const QnnContext_Config_t*>& out_cfg);
    bool AfterCreate();

    const QnnInterface_t* be_{nullptr};
    Qnn_ContextHandle_t ctx_{nullptr};
    Qnn_BackendHandle_t backend_{nullptr};
    Qnn_DeviceHandle_t device_{nullptr};

    std::vector<QnnContext_Config_t> cfg_storage_;
    std::vector<std::unique_ptr<QnnHtpContext_CustomConfig_t>> htp_custom_cfg_;

    bool weight_sharing_{false};
    bool use_multi_contexts_{false};
    uint64_t max_sf_buf_size_{0};

    static inline Qnn_ContextHandle_t sf_handle_{0x0};
    Qnn_ProfileHandle_t profiler_{nullptr};
};