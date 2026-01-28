#include "qnn_sharedbuffer.h"

#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <mutex>

static inline size_t align_up(size_t x, size_t a){
  return (x + (a-1)) & ~(a-1);
}

bool SharedBuffer::ArenaCreate(Arena& a, size_t total_bytes, size_t alignment){
#if !defined(__aarch64__)
    (void) a; (void) total_bytes; (void) alignment;
    return false;
#else
    if (alignment == 0) alignment = 64;

    void* base = AllocMem(total_bytes, alignment);
    if(!base) {
      std::cerr << "Something wrong during shared buffer allocation in ArenaCreate\n";
      return false;
    }

    int fd = MemToFd(base);
    if (fd < 0){
      std::cerr << "Failed to get Fd from shared buffer memory in ArenaCreate\n";
      return false;
    }

    size_t total = GetAllocatedSize(base);
    if(total==0) total = total_bytes;

    a.base = base;
    a.fd = fd;
    a.total = total;
    a.cursor = 0;
    a.alignment = alignment;
    return true;
#endif
}

bool SharedBuffer::ArenaAlloc(Arena& a, size_t bytes, size_t alignment, void** out_ptr, size_t* out_offset){
#if !defined(__aarch64__)
  (void)a; (void) bytes; (void) alignment; (void) out_ptr; (void) out_offset;
  return false;
#else
  if (!a.base || a.fd < 0 || !out_ptr || !out_offset) return false;
  if (alignment == 0) alignment = a.alignment ? a.alignment : 64;

  size_t off = align_up(a.cursor, alignment);
  if (off + bytes > a.total){
    std::cerr << "There is no room to allocate in this arena, total bytes is " << a.total << " but needs more than " << off+bytes << std::endl;
    return false;
  }

  *out_offset = off;
  *out_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(a.base) + off);

  a.cursor = off + bytes;
  return true;
#endif
}

void SharedBuffer::ArenaDestroy(Arena& a){
#if !defined(__aarch64__)
  a = Arena();
#else
  if (a.base) FreeMem(a.base);
  a = Arena();
#endif
}

namespace{
    constexpr uint8_t RPCMEM_HEAP_ID_SYSTEM = 25;
    constexpr uint8_t RPCMEM_DEFAULT_FLAGS = 1;

    static inline intptr_t alignTo(size_t alignment, intptr_t p){
        intptr_t a = static_cast<intptr_t>(alignment);
        return (p % a == 0) ? p : (p + (a - (p % a)));
    }
}

std::mutex SharedBuffer::init_mutex_;

SharedBuffer& SharedBuffer::Instance(){
    std::lock_guard<std::mutex> lk(init_mutex_);
    static SharedBuffer sb;
    if(!sb.initialized_.load()){
#if defined(__aarch64__)
        if(! sb.Load()){
            std::cerr << "[QNN] SharedBuffer: Load Failed\n";
        } else{
            sb.initialized_.store(true);
        }
#else
        sb.initialized_.store(false);
#endif
    }
    return sb;
}

SharedBuffer::~SharedBuffer(){
#if defined(__aarch64__)
    if(initialized_.load()){
        Unload();
        initialized_.store(false);
    }
#endif
}

bool SharedBuffer::Load(){
    lib_cdsp_rpc_ = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib_cdsp_rpc_){
        std::cerr << "[QNN] dlopen(libcdsprpc.so) failed : " << dlerror() << "\n";
        return false;
    }

    rpc_mem_alloc_ = reinterpret_cast<RpcMemAllocFn_t>(dlsym(lib_cdsp_rpc_, "rpcmem_alloc"));
    rpc_mem_free_ = reinterpret_cast<RpcMemFreeFn_t>(dlsym(lib_cdsp_rpc_, "rpcmem_free"));
    rpc_mem_to_fd_ = reinterpret_cast<RpcMemToFdFn_t>(dlsym(lib_cdsp_rpc_, "rpcmem_to_fd"));

    if (!rpc_mem_alloc_ || !rpc_mem_free_ || !rpc_mem_to_fd_){
        std::cerr << "[QNN] dlsym(rpcmem_*) failed : " << dlerror() << "\n";
        dlclose(lib_cdsp_rpc_);
        lib_cdsp_rpc_  = nullptr;
        return false;
    }
    return true;
}

void SharedBuffer::Unload() {
  if (lib_cdsp_rpc_) {
    dlclose(lib_cdsp_rpc_);
    lib_cdsp_rpc_ = nullptr;
  }
  rpc_mem_alloc_ = nullptr;
  rpc_mem_free_ = nullptr;
  rpc_mem_to_fd_ = nullptr;
  restore_map_.clear();
  allocated_size_map_.clear();
}

void* SharedBuffer::AllocMem(size_t bytes, size_t alignment) {
#if !defined(__aarch64__)
  (void)bytes; (void)alignment;
  std::cerr << "[QNN] SharedBuffer::AllocMem noop on non-aarch64\n";
  return nullptr;
#else
  if (!initialized_.load() || !rpc_mem_alloc_) {
    std::cerr << "[QNN] SharedBuffer not initialized\n";
    return nullptr;
  }

  // rpcmem은 내부 alignment도 있지만, executorch처럼 우리가 추가로 aligned ptr을 돌려준다
  int32_t alloc_bytes = static_cast<int32_t>(bytes + alignment);
  void* raw = rpc_mem_alloc_(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, alloc_bytes);
  if (!raw) {
    std::cerr << "[QNN] rpcmem_alloc failed\n";
    return nullptr;
  }

  allocated_size_map_[raw] = static_cast<size_t>(alloc_bytes);

  void* aligned = reinterpret_cast<void*>(alignTo(alignment, reinterpret_cast<intptr_t>(raw)));
  if (!restore_map_.insert({aligned, raw}).second) {
    std::cerr << "[QNN] restore_map insert failed\n";
    rpc_mem_free_(raw);
    allocated_size_map_.erase(raw);
    return nullptr;
  }
  return aligned;
#endif
}

int32_t SharedBuffer::MemToFd(void* buf) {
#if !defined(__aarch64__)
  (void)buf;
  return -1;
#else
  if (!initialized_.load() || !rpc_mem_to_fd_) return -1;
//   // 주의: rpcmem_to_fd는 “raw든 aligned든” 들어오는 ptr이 rpcmem 영역이면 보통 동작하지만,
//   // 안전하게 하려면 raw로 변환해서 넣는 게 좋다.
//   void* raw = buf;
//   auto it = restore_map_.find(buf);
//   if (it != restore_map_.end()) raw = it->second;
//   return rpc_mem_to_fd_(raw);
  return rpc_mem_to_fd_(buf);
#endif
}

void SharedBuffer::FreeMem(void* buf) {
#if !defined(__aarch64__)
  (void)buf;
#else
  if (!initialized_.load() || !rpc_mem_free_) return;

  auto it = restore_map_.find(buf);
  if (it == restore_map_.end()) {
    std::cerr << "[QNN] FreeMem: not an allocated ptr\n";
    return;
  }

  void* raw = it->second;
  rpc_mem_free_(raw);

  restore_map_.erase(it);
  allocated_size_map_.erase(raw);
#endif
}

bool SharedBuffer::IsAllocated(void* buf) const {
#if !defined(__aarch64__)
  (void)buf;
  return false;
#else
  return restore_map_.count(buf) != 0;
#endif
}

size_t SharedBuffer::GetAllocatedSize(void* buf) const {
#if !defined(__aarch64__)
  (void)buf;
  return 0;
#else
//   // buf가 aligned로 들어오면 raw로 바꿔서 size map에서 찾는다
//   void* raw = buf;
//   auto it = restore_map_.find(buf);
//   if (it != restore_map_.end()) raw = it->second;
  void* raw = buf;

  auto it2 = allocated_size_map_.find(raw);
  if (it2 == allocated_size_map_.end()) return 0;
  return it2->second;
#endif
}