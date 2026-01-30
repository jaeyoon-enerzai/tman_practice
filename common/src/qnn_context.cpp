#include "qnn_context.h"
#include "HTP/QnnHtpContext.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnProfile.h"

static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* what) {
  if (err != QNN_SUCCESS) {
    std::cerr << "[QNN] " << what << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
    return false;
  }
  return true;
}

// ---- 내부: context config 구성 ----
// Device 코드랑 동일한 패턴: cfg_storage_ + htp_custom_cfg_ 보관 후
// out_cfg에는 const QnnContext_Config_t* 포인터 배열을 넘김(마지막 nullptr)
bool QnnContextRuntime::MakeConfig(std::vector<const QnnContext_Config_t*>& out_cfg){
  out_cfg.clear();
  cfg_storage_.clear();
  htp_custom_cfg_.clear();

  auto alloc_htp_custom = [&]()-> QnnHtpContext_CustomConfig_t* {
    htp_custom_cfg_.emplace_back(std::make_unique<QnnHtpContext_CustomConfig_t>());
    std::memset(htp_custom_cfg_.back().get(), 0, sizeof(QnnHtpContext_CustomConfig_t));
    htp_custom_cfg_.back()->option = QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN;
    return htp_custom_cfg_.back().get();
  };

  std::vector<QnnContext_CustomConfig_t> custom_cfgs;

#if defined(__aarch64__)
  QnnHtpContext_CustomConfig_t* p;
  if (use_multi_contexts_ && max_sf_buf_size_ != 0){
    p = alloc_htp_custom();
    p->option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    QnnHtpContext_GroupRegistration_t group_info;
    group_info.firstGroupHandle = sf_handle_;
    group_info.maxSpillFillBuffer = max_sf_buf_size_;
    p->groupRegistration = group_info;
    custom_cfgs.push_back(static_cast<QnnContext_CustomConfig_t>(p));
  }
  // heap usage profling
  p = alloc_htp_custom();
  p->option = QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED;
  p->dspMemoryProfilingEnabled = true;
  custom_cfgs.push_back(static_cast<QnnContext_CustomConfig_t>(p));
#else
  if (weight_sharing_){
    QnnHtpContext_CustomConfig_t* p = alloc_htp_custom();
    p->option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    p->weightSharingEnabled = true;
    custom_cfgs.push_back(static_cast<QnnContext_CustomConfig_t>(p));
  }
#endif

  cfg_storage_.resize(custom_cfgs.size());
  out_cfg.reserve(custom_cfgs.size() + 1);

  for(size_t i=0; i < custom_cfgs.size(); ++i){
    cfg_storage_[i].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    cfg_storage_[i].customConfig = custom_cfgs[i];
    out_cfg.push_back(&cfg_storage_[i]);
  }
  out_cfg.push_back(nullptr);
  return true;
}

bool QnnContextRuntime::AfterCreate(){
#if defined (__aarch64__)
  if (sf_handle_ == 0x0 && ctx_ != nullptr){
    sf_handle_ = ctx_;
  }
#endif
  return true;
}

bool QnnContextRuntime::Create(const QnnInterface_t* be,
                                Qnn_BackendHandle_t backend_handle,
                                Qnn_DeviceHandle_t device_handle
                              ){
    if(!be || !backend_handle){
        std::cerr << "[QNN] Context Create: invalid backend\n";
        return false;
    }
    be_ = be;
    backend_ = backend_handle;
    device_ = device_handle;

    if(ctx_) return true;

    auto& api = be_->QNN_INTERFACE_VER_NAME;

    std::vector<const QnnContext_Config_t*> cfg;
    if(!MakeConfig(cfg)){
      std::cerr << "[QNN] Context MakeConfig Failed\n";
      return false;
    }

    const QnnContext_Config_t** cfg_ptr = cfg.empty() ? nullptr : cfg.data();
    Qnn_ErrorHandle_t err = api.contextCreate(backend_handle,
                                    device_handle,
                                cfg_ptr,
                            &ctx_);
    if (!CheckQnnOk(err, "contextCreate")) return false;

    if (!AfterCreate()) return false;
    return true;
}

bool QnnContextRuntime::CreateFromBinary(const QnnInterface_t* be,
                    Qnn_BackendHandle_t backend_handle,
                    Qnn_DeviceHandle_t device_handle,
                    Qnn_ProfileHandle_t profileHandle,
                    const uint8_t* ctx_bin,
                    uint32_t ctx_bin_bytes){
  if(!be || !backend_handle){
      std::cerr << "[QNN] Context CreateFromBinary: invalid backend\n";
      return false;
  }
  if (!ctx_bin || ctx_bin_bytes == 0){
      std::cerr << "[QNN] Context CreateFromBinary: invalid binary\n";
      return false;
  }
  be_ = be;
  backend_ = backend_handle;
  device_ = device_handle;
  profiler_ = profileHandle;

  if(ctx_) return true;

  auto& api = be_->QNN_INTERFACE_VER_NAME;

  std::vector<const QnnContext_Config_t*> cfg;
  if(!MakeConfig(cfg)){
    std::cerr << "[QNN] Context MakeConfig Failed\n";
    return false;
  }

  const QnnContext_Config_t** cfg_ptr = cfg.empty() ? nullptr : cfg.data();
#if defined (__aarch64__)
  Qnn_ErrorHandle_t err = api.contextCreateFromBinary(backend_handle, device_handle, cfg_ptr, ctx_bin, ctx_bin_bytes, &ctx_, /*profile=*/profileHandle);
  
  const QnnProfile_EventId_t* events;
  uint32_t numEvents;
  be_->QNN_INTERFACE_VER_NAME.profileGetEvents(profileHandle, &events, &numEvents);
  for(uint32_t i=0; i < numEvents; ++i){
    QnnProfile_EventData_t eventData;
    be_->QNN_INTERFACE_VER_NAME.profileGetEventData(events[i], &eventData);
    if (strcmp(eventData.identifier, "DSP:before_context_created") == 0){
      std::cout << "total DspHeap Usage Before Context Created : " << eventData.value << std::endl;
    }
  }
  if(!CheckQnnOk(err, "contextCreateFromBinary")) return false;
#else
  std::cerr << "[QNN] CreateFromBinary is intended for aarch64 runtime only\n";
  return false;
#endif

  if(!AfterCreate()) return false;
  return true;
}

bool QnnContextRuntime::GetBinary(std::vector<uint8_t>& out_blob){
    if (!be_ || !ctx_) return false;

    auto& api = be_->QNN_INTERFACE_VER_NAME;

    Qnn_ContextBinarySize_t bin_size = 0;
    if (!CheckQnnOk(api.contextGetBinarySize(ctx_, &bin_size), "contextGetBinarySize")) return false;

    out_blob.resize(static_cast<size_t>(bin_size));
    Qnn_ContextBinarySize_t bytes_written = 0;

    if(!CheckQnnOk(api.contextGetBinary(
        ctx_,
        out_blob.data(),
        bin_size,
        &bytes_written
    ), "contextGetBinary")) return false;

    if (bytes_written > bin_size) {
        std::cerr << "[QNN] contextGetBinary wrote too much: " << bytes_written
                << " > " << bin_size << "\n";
        return false;
    }
    out_blob.resize(static_cast<size_t>(bytes_written));
    return true;
}

void QnnContextRuntime::Destroy() {
  if (!be_ || !ctx_) return;

  auto& api = be_->QNN_INTERFACE_VER_NAME;

  // ExecuTorch가 쓰는 형태: contextFree(handle, profile=nullptr)
  // 네 SDK가 인자를 2개 요구하면 이게 맞고,
  // 1개만 요구하면 두 번째 인자를 지우면 됨.
  (void)CheckQnnOk(api.contextFree(ctx_, /*profile=*/profiler_), "contextFree");

  if(profiler_){
    const QnnProfile_EventId_t* events;
    uint32_t numEvents;
    be_->QNN_INTERFACE_VER_NAME.profileGetEvents(profiler_, &events, &numEvents);
    for(uint32_t i=0; i < numEvents; ++i){
      QnnProfile_EventData_t eventData;
      be_->QNN_INTERFACE_VER_NAME.profileGetEventData(events[i], &eventData);
      if (strcmp(eventData.identifier, "DSP:after_context_freed") == 0){
        std::cout << "total DspHeap Usage After Context Created : " << eventData.value << std::endl;
      }
    }
  }

  ctx_ = nullptr;
  profiler_ = nullptr;

}