#include "qnn_device.h"
#include "QnnTypes.h"
#include "qnn_option.h"
#include "qnn_platform.h"
#include "HTP/QnnHtpDevice.h"
#include "QnnCommon.h"
#include "QnnDevice.h"
#include "QnnInterface.h"
#include <memory>

static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* what) {
  if (err != QNN_SUCCESS) {
    std::cerr << "[QNN] " << what << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
    return false;
  }
  return true;
}

QnnDeviceRuntime::~QnnDeviceRuntime() {
  Destroy();
}

bool QnnDeviceRuntime::MakeConfig(std::vector<const QnnDevice_Config_t*>& out_cfg){
    // TODO - add configuration
    out_cfg.clear();
    cfg_storage_.clear();
    htp_custom_cfg_.clear();

  auto alloc_htp_custom = [&]() -> QnnHtpDevice_CustomConfig_t* {
    htp_custom_cfg_.emplace_back(std::make_unique<QnnHtpDevice_CustomConfig_t>());
    std::memset(htp_custom_cfg_.back().get(), 0, sizeof(QnnHtpDevice_CustomConfig_t));
    htp_custom_cfg_.back()->option = QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN;
    return htp_custom_cfg_.back().get();
  };

  // HTP CUSTOM
  QnnHtpDevice_CustomConfig_t* p = nullptr;
  std::vector<QnnDevice_CustomConfig_t> device_custom_config = {};

  p = alloc_htp_custom();
  p->option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
  p->arch.deviceId = 0;
  // p->arch.arch = QNN_HTP_DEVICE_ARCH_V68;
  p->arch.arch = static_cast<QnnHtpDevice_Arch_t>(ARCH);
  device_custom_config.push_back(static_cast<QnnDevice_CustomConfig_t>(p));


  // p = alloc_htp_custom();
  // p->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  // p->socModel = static_cast<uint32_t>(SOC);
  // // p->socModel = QNN_SOC_MODEL_SM8850;
  // // p->socModel = QNN_SOC_MODEL_SM8750;
  // device_custom_config.push_back(static_cast<QnnDevice_CustomConfig_t>(p));
  

  switch(PD_SESSION){
    case QnnHtpPdSession::kHtpSignedPd: {
      p = alloc_htp_custom();
      p->option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
      p->useSignedProcessDomain.useSignedProcessDomain = true;
      p->useSignedProcessDomain.deviceId = 0;
      device_custom_config.push_back(static_cast<QnnDevice_CustomConfig_t>(p));
      break;
    }
    case QnnHtpPdSession::kHtpUnsignedPd:
    default:
      break;
  }
  
  // see aarch64/HtpDevicePlatformInfoConfig.cp
  const std::vector<QnnDevice_PlatformInfo_t*>& device_platform_info = htp_device_platform_info_config_->CreateDevicePlatformInfo();

  uint32_t num_custom_configs = device_platform_info.size() + device_custom_config.size();

  cfg_storage_.resize(num_custom_configs);
  // +1 for null terminated
  out_cfg.reserve(num_custom_configs +1);

  for(std::size_t i=0; i < device_custom_config.size(); ++i){
    cfg_storage_[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    cfg_storage_[i].customConfig = device_custom_config[i];
    out_cfg.push_back(&cfg_storage_[i]);
  }

  if(!device_platform_info.empty()){
    // Below codes use `Device_config_[device_custom_config.size()]` which imply
    // the length of platform info can only be 1.
    if(device_platform_info.size() != 1u){
      std::cerr << "Error! Device platform info size != 1, got " << device_platform_info.size();
    }
    cfg_storage_[device_custom_config.size()].option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    cfg_storage_[device_custom_config.size()].hardwareInfo = device_platform_info.back();
    out_cfg.push_back(&cfg_storage_[device_custom_config.size()]);
  }

  // Do we need "null-terminated"?
  out_cfg.push_back(nullptr);
  return true;
}

bool QnnDeviceRuntime::AfterCreateDevice(){
    // TODO - HTP perf vote / rpc polling / platform info
    return true;
}

bool QnnDeviceRuntime::Create(const QnnInterface_t* be_iface, Qnn_LogHandle_t logger_handle){
    if (!be_iface){
        std::cerr << "[QNN] Device Create: be_iface is null\n";
        return false;
    }
    be_ = be_iface;
    logger_handle_ = logger_handle;
    
    if (device_handle_) return true;

    std::vector<const QnnDevice_Config_t*> cfg;
    if (!MakeConfig(cfg)){
        std::cerr << "[QNN] Device MakeConfig failed\n";
        return false;
    }

      std::cout << "number of device config is " << cfg.size() << std::endl;


    const QnnDevice_Config_t* const* cfg_ptr = cfg.empty() ? nullptr : cfg.data();

    auto& api = be_->QNN_INTERFACE_VER_NAME;

    // drop only the top-level const on the pointer-to-pointer
    const QnnDevice_Config_t** cfg_pp =
        cfg_ptr ? const_cast<const QnnDevice_Config_t**>(cfg_ptr) : nullptr;

    if (!CheckQnnOk(api.deviceCreate(logger_handle_, cfg_pp, &device_handle_), "deviceCreate")) {
    return false;
    }

    if (!AfterCreateDevice()){
        std::cerr << "[QNN] Device AfterCreateDevice failed\n";
        return false;
    }
    return true;
}

void QnnDeviceRuntime::Destroy() {
  if (!be_ || !device_handle_) return;

  auto& api = be_->QNN_INTERFACE_VER_NAME;
  (void)CheckQnnOk(api.deviceFree(device_handle_), "deviceFree");
  device_handle_ = nullptr;
}