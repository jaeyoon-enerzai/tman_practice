#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include "QnnTypes.h"
#include "QnnInterface.h"
#include "QnnCommon.h"

#include "HTP/QnnHtpMem.h"
#include "qnn_tensor.h"
#include "qnn_context.h"

class QnnConextRuntime;

class QnnMemManagerRuntime{
    public:
    QnnMemManagerRuntime() = default;
    ~QnnMemManagerRuntime() { DeRegisterAll(); }

    QnnMemManagerRuntime(const QnnMemManagerRuntime&) = delete;
    QnnMemManagerRuntime& operator=(const QnnMemManagerRuntime&) = delete;

    bool Init(const QnnInterface_t* be_iface, QnnContextRuntime* ctx);

    bool RegisterIon(Qnn_Tensor_t& tensor_meta, // rank/dims/dtype
                    int32_t mem_fd,
                    void* mem_ptr,
                    Qnn_MemHandle_t* out_handle
                    );
    bool RegisterHtpSharedBufferCustom(
        Qnn_Tensor_t& tensor_meta, // rank/dims/dtype
        int32_t mem_fd,
        void* mem_ptr,
        size_t total_custom_mem_size,
        size_t tensor_offset,
        Qnn_MemHandle_t* out_handle
    );

    bool SetTensorMemHandle(Qnn_Tensor_t& t, Qnn_MemHandle_t handle);

    bool IsReigstered(Qnn_MemHandle_t handle, void* mem_ptr) const;

    void DeRegisterAll();

    private:
        const QnnInterface_t* be_{nullptr};
        QnnContextRuntime* ctx_{nullptr};

    std::unordered_map<Qnn_MemHandle_t, void*> registered_;

};