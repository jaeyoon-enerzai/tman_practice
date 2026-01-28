#include "qnn_mem_manager.h"
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "qnn_context.h"
#include "qnn_tensor.h"
#include <cstddef>

static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* what){
    if (err != QNN_SUCCESS){
        std::cerr << "[QNN] " << what << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
        return false;
    }
    return true;
}

bool QnnMemManagerRuntime::Init(const QnnInterface_t* be_iface, QnnContextRuntime* ctx){
    if (!be_iface || !ctx || !ctx->IsValid()){
        std::cerr << "[QNN] MemManager Init: invalid be/ctx\n";
        return false;
    }
    be_ = be_iface;
    ctx_ = ctx;
    return true;
}

bool QnnMemManagerRuntime::IsReigstered(Qnn_MemHandle_t handle, void* mem_ptr) const{
    auto it = registered_.find(handle);
    if (it != registered_.end()){
        return it->second == mem_ptr;
    }
    return false;
}

bool QnnMemManagerRuntime::RegisterIon(Qnn_Tensor_t& tensor_meta, int32_t mem_fd,
                                    void* mem_ptr, Qnn_MemHandle_t* out_handle){
#if !defined(__aarch64__)
    (void) tensor_meta; (void) mem_fd; (void) mem_ptr;
    if(out_handle) *out_handle = nullptr;
    std::cerr << "[QNN] RegisterIon: noop on non-aarch64\n";
    return false;
#else
    if (!be_ || !ctx_ || !ctx_->IsValid() || !out_handle) return false;

    auto& api = be_->QNN_INTERFACE_VER_NAME;

    Qnn_MemDescriptor_t desc{};
    desc.memShape.numDim = QNN_TENSOR_VER_PTR(tensor_meta)->rank;
    desc.memShape.dimSize = QNN_TENSOR_VER_PTR(tensor_meta)->dimensions;
    desc.memShape.shapeConfig = nullptr;
    desc.dataType = QNN_TENSOR_VER_PTR(tensor_meta)->dataType;
    desc.memType = QNN_MEM_TYPE_ION;
    desc.ionInfo.fd = mem_fd;

    Qnn_MemHandle_t handle = nullptr;
    auto err = api.memRegister(ctx_->Handle(), &desc, /*numDescriptors=*/1, &handle);
    if(!CheckQnnOk(err, "memRegister(ION)")) return false;

    SetTensorMemHandle(tensor_meta, handle);
    registered_.insert({handle, mem_ptr});
    *out_handle = handle;
    return true;
#endif
}

bool QnnMemManagerRuntime::RegisterHtpSharedBufferCustom(
    Qnn_Tensor_t& tensor_meta,
    int32_t mem_fd,
    void* mem_ptr,
    size_t total_custom_mem_size,
    size_t tensor_offset,
    Qnn_MemHandle_t* out_handle
){
#if !defined (__aarch64__)
    (void) tensor_meta; (void) mem_fd; (void) mem_ptr; (void) total_custom_mem_size; (void) tensor_offset;
    if(out_handle) *out_handle = nullptr;
    std::cerr << "[QNN] RegisterCustom: noop on non-aarch64\n";
    return false;
#else
    if (!be_ || !ctx_ || !ctx_->IsValid() || !out_handle) return false;

    auto &api = be_->QNN_INTERFACE_VER_NAME;

    Qnn_MemDescriptor_t desc{};
    desc.memShape.numDim = QNN_TENSOR_VER_PTR(tensor_meta)->rank;
    desc.memShape.dimSize = QNN_TENSOR_VER_PTR(tensor_meta)->dimensions;
    desc.memShape.shapeConfig = nullptr;
    desc.dataType = QNN_TENSOR_VER_PTR(tensor_meta)->dataType;
    desc.memType = QNN_MEM_TYPE_CUSTOM;

    QnnMemHtp_Descriptor_t htp_desc{};
    htp_desc.type = QNN_HTP_MEM_SHARED_BUFFER;
    htp_desc.size = total_custom_mem_size;

    QnnHtpMem_SharedBufferConfig_t sb_cfg{};
    sb_cfg.fd = mem_fd;
    sb_cfg.offset = tensor_offset;
    htp_desc.sharedBufferConfig = sb_cfg;

    desc.customInfo = &htp_desc; // static cast < Qnn_MemInfoCustom_t> 안해도 되나?

    Qnn_MemHandle_t handle = nullptr;
    auto err = api.memRegister(ctx_->Handle(), &desc, 1, &handle);
    if(!CheckQnnOk(err, "memRegister(CUSTOM/HTP_SHARED_BUFFER)")) return false;

    SetTensorMemHandle(tensor_meta, handle);
    registered_.insert({handle, mem_ptr});
    *out_handle = handle;
    return true;
#endif
}

/* TODO
bool QnnMemManagerRuntime::PreRegisterHtpSharedBufferCustom (see PreRegisterCustomMemHandle in QnnMemManager.cpp)
*/

bool QnnMemManagerRuntime::SetTensorMemHandle(Qnn_Tensor_t& t, Qnn_MemHandle_t handle){
    if (!handle) return false;
    QNN_TENSOR_VER_PTR(t)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    QNN_TENSOR_VER_PTR(t)->memHandle = handle;
    // for safe
    QNN_TENSOR_VER_PTR(t)->clientBuf.data = nullptr;
    QNN_TENSOR_VER_PTR(t)->clientBuf.dataSize = 0;
    return true;
}

void QnnMemManagerRuntime::DeRegisterAll(){
#if !defined(__aarch64__)
    registered_.clear();
    return;
#else
    if(!be_ || registered_.empty()) return;
    auto& api = be_->QNN_INTERFACE_VER_NAME;

    for(auto& kv: registered_){
        Qnn_MemHandle_t h= kv.first;
        auto err = api.memDeRegister(&h, 1);
        (void)CheckQnnOk(err, "memDeRegister");
    }
    registered_.clear();
#endif
}