#pragma once
#include <cstdint>
#include <iostream>
#include <vector>

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnContext.h"

class QnnContextRuntime{
    public:
    QnnContextRuntime() = default;
    ~QnnContextRuntime() { Destroy();}

    QnnContextRuntime(const QnnContextRuntime&) = delete;
    QnnContextRuntime& operator=(const QnnContextRuntime&) = delete;

    bool Create(const QnnInterface_t* be,
                Qnn_BackendHandle_t backend_handle,
                Qnn_DeviceHandle_t device_handle_,
                const QnnContext_Config_t** cfg = nullptr);

    bool GetBinary(std::vector<uint8_t>& out_blob);

    void Destroy();

    Qnn_ContextHandle_t Handle() const {return ctx_;}
    bool IsValid() const { return ctx_ != nullptr;}

    private:
    const QnnInterface_t* be_{nullptr};
    Qnn_ContextHandle_t ctx_{nullptr};
    Qnn_BackendHandle_t backend_{nullptr};
    Qnn_DeviceHandle_t device_{nullptr};
};