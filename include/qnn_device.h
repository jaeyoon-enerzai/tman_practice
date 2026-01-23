#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <cstring>
#include <iostream>

#include "QnnCommon.h"
#include "QnnDevice.h"
#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "qnn_platform.h"

class QnnDeviceRuntime{
    public:
    QnnDeviceRuntime(){
        htp_device_platform_info_config_ = std::make_unique<QnnDevicePlatformInfoRuntime>(vtcmSize, PD_SESSION, SOC, ARCH);
    };
    ~QnnDeviceRuntime();

    QnnDeviceRuntime(const QnnDeviceRuntime&) = delete;
    QnnDeviceRuntime& operator=(const QnnDeviceRuntime&) = delete;

    bool Create(const QnnInterface_t* be_iface,
                    Qnn_LogHandle_t logger_handle=nullptr);
    void Destroy();

    Qnn_DeviceHandle_t Handle() const { return device_handle_;}
    bool IsValid() const {return device_handle_ != nullptr;}

    protected:
    virtual bool MakeConfig(std::vector<const QnnDevice_Config_t*>& out_cfg);
    virtual bool AfterCreateDevice(); // something like perf vote

    private:

// const - TODO : parse from provided config
QnnHtpPdSession PD_SESSION=QnnHtpPdSession::kHtpUnsignedPd;
// QcomChipset SOC = QcomChipset::QCS6490;
QcomChipset SOC = QcomChipset::SM8750;
// uint32_t vtcmSize = 2;
uint32_t vtcmSize = 8;
// HtpArch ARCH = HtpArch::V68;
HtpArch ARCH = HtpArch::V73;


    const QnnInterface_t* be_{nullptr};
    Qnn_LogHandle_t logger_handle_{nullptr};
    Qnn_DeviceHandle_t device_handle_{nullptr};

    std::unique_ptr<QnnDevicePlatformInfoRuntime> htp_device_platform_info_config_;

    std::vector<QnnDevice_Config_t> cfg_storage_;    
    std::vector<std::unique_ptr<QnnHtpDevice_CustomConfig_t>> htp_custom_cfg_;
};