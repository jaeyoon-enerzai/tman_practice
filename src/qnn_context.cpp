#include "qnn_context.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnInterface.h"
#include "QnnTypes.h"

static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* what) {
  if (err != QNN_SUCCESS) {
    std::cerr << "[QNN] " << what << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
    return false;
  }
  return true;
}

bool QnnContextRuntime::Create(const QnnInterface_t* be,
                                Qnn_BackendHandle_t backend_handle,
                                Qnn_DeviceHandle_t device_handle,
                              const QnnContext_Config_t** cfg){
    if(!be || !backend_handle){
        std::cerr << "[QNN] Context Create: invalid backend\n";
        return false;
    }
    be_ = be;
    backend_ = backend_handle;
    device_ = device_handle;

    if(ctx_) return true;

    auto& api = be_->QNN_INTERFACE_VER_NAME;

    // const QnnContext_Config_t** cfg = nullptr;

    // TODO - Later, in the runtime we should get context from already created context binary
    Qnn_ErrorHandle_t err = api.contextCreate(backend_handle,
                                    device_handle,
                                cfg,
                            &ctx_);
    if (!CheckQnnOk(err, "contextCreate")) return false;
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
  (void)CheckQnnOk(api.contextFree(ctx_, /*profile=*/nullptr), "contextFree");
  ctx_ = nullptr;
}