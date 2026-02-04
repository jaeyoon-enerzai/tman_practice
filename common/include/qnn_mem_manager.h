#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include "QnnTypes.h"
#include "QnnInterface.h"
#include "QnnCommon.h"

#include "HTP/QnnHtpMem.h"
#include "qnn_sharedbuffer.h"
#include "qnn_tensor.h"
#include "qnn_context.h"

class QnnConextRuntime;
class SharedBuffer;

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

    // out_ptr : host에서 접근할 pointer
    // out_handle : memRegister 결과 핸들
    bool RegisterTensorInSharedArena(
        SharedBuffer& sb, SharedBuffer::Arena& arena,
        Qnn_Tensor_t& tensor_meta, size_t tensor_bytes,
        size_t alignment, void** out_ptr, Qnn_MemHandle_t* out_handle, size_t* out_offset = nullptr);

    private:
        const QnnInterface_t* be_{nullptr};
        QnnContextRuntime* ctx_{nullptr};

    std::unordered_map<Qnn_MemHandle_t, void*> registered_;
    std::vector<std::unique_ptr<QnnMemHtp_Descriptor_t>> htp_desc_storage_;
    std::unordered_map<uint64_t, Qnn_MemHandle_t> sb_handle_by_key_;

};