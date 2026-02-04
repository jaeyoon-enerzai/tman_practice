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
    if (tensor_meta.version == QNN_TENSOR_VERSION_2){
        desc.memShape.numDim = QNN_TENSOR_VER_PTR(tensor_meta)->rank;
        desc.memShape.dimSize = QNN_TENSOR_VER_PTR(tensor_meta)->dimensions;
        desc.memShape.shapeConfig = nullptr;
        desc.dataType = QNN_TENSOR_VER_PTR(tensor_meta)->dataType;
        desc.memType = QNN_MEM_TYPE_ION;
        desc.ionInfo.fd = mem_fd;
    }else if(tensor_meta.version == QNN_TENSOR_VERSION_1){
        desc.memShape.numDim = tensor_meta.v1.rank;
        desc.memShape.dimSize = tensor_meta.v1.dimensions;
        desc.memShape.shapeConfig = nullptr;
        desc.dataType = tensor_meta.v1.dataType;
        desc.memType = QNN_MEM_TYPE_ION;
        desc.ionInfo.fd = mem_fd;
    }else{
        std::cerr << "unknown tensor version " << tensor_meta.version << "\n";
    }
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
    if (!be_ || !ctx_ || !ctx_->IsValid() || !out_handle) {
        return false;
    }
    auto &api = be_->QNN_INTERFACE_VER_NAME;

    Qnn_MemDescriptor_t desc{};
    if (tensor_meta.version == QNN_TENSOR_VERSION_2){
        desc.memShape.numDim = QNN_TENSOR_VER_PTR(tensor_meta)->rank;
        desc.memShape.dimSize = QNN_TENSOR_VER_PTR(tensor_meta)->dimensions;
        desc.memShape.shapeConfig = nullptr;
        desc.dataType = QNN_TENSOR_VER_PTR(tensor_meta)->dataType;
        desc.memType = QNN_MEM_TYPE_CUSTOM;
    } else if(tensor_meta.version == QNN_TENSOR_VERSION_1){
        desc.memShape.numDim = tensor_meta.v1.rank;
        desc.memShape.dimSize = tensor_meta.v1.dimensions;
        desc.memShape.shapeConfig = nullptr;
        desc.dataType = tensor_meta.v1.dataType;
        desc.memType = QNN_MEM_TYPE_CUSTOM;
    } else{
        std::cerr << "unknown tensor version : " << tensor_meta.version << "\n";
        return false;
    }

    desc.ionInfo.fd = mem_fd;

    auto htp_desc = std::make_unique<QnnMemHtp_Descriptor_t>();
    std::memset(htp_desc.get(), 0, sizeof(QnnMemHtp_Descriptor_t));

    htp_desc->type = QNN_HTP_MEM_SHARED_BUFFER;
    htp_desc->size = total_custom_mem_size;

    QnnHtpMem_SharedBufferConfig_t sb_cfg{};
    sb_cfg.fd = mem_fd;
    sb_cfg.offset = tensor_offset;
    htp_desc->sharedBufferConfig = sb_cfg;

    desc.customInfo = reinterpret_cast<Qnn_MemInfoCustom_t>(htp_desc.get()); // static cast < Qnn_MemInfoCustom_t> 안해도 되나?
    htp_desc_storage_.push_back(std::move(htp_desc));

    Qnn_MemHandle_t handle = nullptr;
    auto err = api.memRegister(ctx_->Handle(), &desc, 1, &handle);
    if(!CheckQnnOk(err, "memRegister(CUSTOM/HTP_SHARED_BUFFER)")) return false;

    if(!SetTensorMemHandle(tensor_meta, handle)){
        std::cerr << "Failed to set mem Handle\n";
        return false;
    };
    registered_.insert({handle, mem_ptr});
    *out_handle = handle;
    return true;
#endif
}

/* TODO
bool QnnMemManagerRuntime::PreRegisterHtpSharedBufferCustom (see PreRegisterCustomMemHandle in QnnMemManager.cpp)
*/

bool QnnMemManagerRuntime::SetTensorMemHandle(Qnn_Tensor_t& t, Qnn_MemHandle_t handle){
    if (handle == nullptr) return false;

    if (t.version == QNN_TENSOR_VERSION_2){
        QNN_TENSOR_VER_PTR(t)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        QNN_TENSOR_VER_PTR(t)->memHandle = handle;
        return true;
    } else if(t.version == QNN_TENSOR_VERSION_1){
        t.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        t.v1.memHandle = handle;
        return true;
    } else{
        std::cerr << "[QNN] tensor version is weird " << t.version << "\n";
        return false;
    }
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
    htp_desc_storage_.clear();
#endif
}

static inline uint64_t MakeKey(int fd, size_t off) {
  return (uint64_t(uint32_t(fd)) << 32) | uint64_t(uint32_t(off));
}

bool QnnMemManagerRuntime::RegisterTensorInSharedArena(
        SharedBuffer& sb, SharedBuffer::Arena& arena,
        Qnn_Tensor_t& tensor_meta, size_t tensor_bytes,
        size_t alignment, void** out_ptr, Qnn_MemHandle_t* out_handle, size_t* out_offset){
#if !defined(__aarch64__)
  (void)sb; (void)arena; (void)out_handle; (void)out_offset;
  return false;
#else
  if (!out_ptr || !out_handle) return false;
  *out_ptr = nullptr;
  *out_handle = nullptr;

  void* ptr = nullptr;
  size_t off = 0;
  if (!sb.ArenaAlloc(arena, tensor_bytes, alignment, &ptr, &off)){
    std::cerr << "[QNN] Arena Alloc failed. bytes = " << tensor_bytes << "\n";
    return false;
  }

  const uint64_t key = MakeKey(arena.fd, off);
  auto it = sb_handle_by_key_.find(key);
  if(it != sb_handle_by_key_.end()){
    Qnn_MemHandle_t h = it->second;
    SetTensorMemHandle(tensor_meta, h);
    *out_ptr = ptr;
    *out_handle = h;
    if(out_offset) *out_offset = off;
    return true;
  }

  Qnn_MemHandle_t h = nullptr;
  if (!RegisterHtpSharedBufferCustom(
    tensor_meta, arena.fd, ptr, arena.total, off, &h
  )){
    std::cerr << "[QNN] RegisterHtpSharedBufferCustom failed\n";
    return false;
  }

  sb_handle_by_key_[key] = h;

  *out_ptr = ptr;
  *out_handle = h;
  if(out_offset) *out_offset = off;
  return true;
#endif
}