#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

using RpcMemAllocFn_t = void* (*)(int, uint32_t, int);
using RpcMemFreeFn_t = void(*)(void*);
using RpcMemToFdFn_t = int (*)(void*);

class SharedBuffer{
    public:
    SharedBuffer(const SharedBuffer&) = delete;
    SharedBuffer& operator=(const SharedBuffer&) = delete;
    ~SharedBuffer();

    static SharedBuffer& Instance();

    void* AllocMem(size_t bytes, size_t alignment);
    int32_t MemToFd(void* buf);
    void FreeMem(void* buf);
    bool IsAllocated(void* buf) const;
    size_t GetAllocatedSize(void* buf) const;

    struct Arena{
        void* base{nullptr};   // AllocMem이 반환한 aligned base ptr
        int fd{-1};            // base로부터 얻은 fd
        size_t total{0};       // 실제 allocated total bytes
        size_t cursor{0};      // 다음 slice 위치
        size_t alignment{64};  // default align
    };

    // Alloc one large shared buffer chunk + get fd
    bool ArenaCreate(Arena& a, size_t total_bytes, size_t alignment);

    // Allocate bytes inside arena to each slice
    bool ArenaAlloc(Arena& a, size_t bytes, size_t alignment, void** out_ptr, size_t* out_offset);

    // Destroy arena from base
    void ArenaDestroy(Arena& a);

    private:
    SharedBuffer() = default;

    bool Load();
    void Unload();

    static std::mutex init_mutex_;
    std::atomic_bool initialized_{false};

    void* lib_cdsp_rpc_;
    RpcMemAllocFn_t rpc_mem_alloc_{nullptr};
    RpcMemFreeFn_t rpc_mem_free_{nullptr};
    RpcMemToFdFn_t rpc_mem_to_fd_{nullptr};

    std::unordered_map<void*, void*> restore_map_;
    std::unordered_map<void*, size_t> allocated_size_map_;

    // TODO
    // std::unordered_map<void*, void*> tensor_addr_to_custom_mem_;
    // std::unordered_set<CustomMemTensorInfo> custom_mem_tensor_info_set_;
};