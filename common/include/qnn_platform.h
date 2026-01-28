#pragma once
#include <memory>
#include <vector>
#include "qnn_option.h"

#include "HTP/QnnHtpDevice.h"
#include "QnnDevice.h"

class QnnDevicePlatformInfoRuntime{
    public:
        QnnDevicePlatformInfoRuntime(uint32_t vtcmSize,
                                    QnnHtpPdSession PD_SESSION,
                                    QcomChipset SOC,
                                    HtpArch ARCH):
        vtcmSize_(vtcmSize), PD_SESSION_(PD_SESSION), SOC_(SOC), ARCH_(ARCH) {}
        std::vector<QnnDevice_PlatformInfo_t*> CreateDevicePlatformInfo();

    private:
        // TODO - config로 바꾸기
        uint32_t vtcmSize_;
        QnnHtpPdSession PD_SESSION_;
        QcomChipset SOC_;
        HtpArch ARCH_;

        QnnDevice_PlatformInfo_t* AllocDevicePlatformInfo() {
            htp_platform_info_.emplace_back(
                std::make_unique<QnnDevice_PlatformInfo_t>());
            htp_platform_info_.back()->version =
                QNN_DEVICE_PLATFORM_INFO_VERSION_UNDEFINED;
            return htp_platform_info_.back().get();
        }

        QnnDevice_HardwareDeviceInfo_t* AllocHwDeviceInfo() {
            htp_hw_device_info_.emplace_back(
                std::make_unique<QnnDevice_HardwareDeviceInfo_t>());
            htp_hw_device_info_.back()->version =
                QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_UNDEFINED;
            return htp_hw_device_info_.back().get();
        }

        QnnDevice_CoreInfo_t* AllocCoreInfo() {
            htp_core_info_.emplace_back(std::make_unique<QnnDevice_CoreInfo_t>());
            htp_core_info_.back()->version = QNN_DEVICE_CORE_INFO_VERSION_UNDEFINED;
            return htp_core_info_.back().get();
        }

        QnnHtpDevice_DeviceInfoExtension_t* AllocDeviceInfoExtension() {
            htp_device_info_extension_.emplace_back(
                std::make_unique<QnnHtpDevice_DeviceInfoExtension_t>());
            htp_device_info_extension_.back()->devType = QNN_HTP_DEVICE_TYPE_UNKNOWN;
            return htp_device_info_extension_.back().get();
        }

        std::vector<std::unique_ptr<QnnDevice_PlatformInfo_t>> htp_platform_info_;
        std::vector<std::unique_ptr<QnnDevice_HardwareDeviceInfo_t>>
            htp_hw_device_info_;
        std::vector<std::unique_ptr<QnnDevice_CoreInfo_t>> htp_core_info_;
        std::vector<std::unique_ptr<QnnHtpDevice_DeviceInfoExtension_t>>
            htp_device_info_extension_;
};