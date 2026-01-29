#pragma once
#include <array>
#include <cstdint>
#include <iostream>

#include "QnnInterface.h"
#include "QnnProfile.h"
#include "QnnCommon.h"

enum class QnnProfileLevel{
    Off = 0,
    Basic,
    Detailed,
    Optrace
};

class QnnProfilerRuntime{
    public:
    QnnProfilerRuntime() = default;
    ~QnnProfilerRuntime() {Destory();}

    QnnProfilerRuntime(const QnnProfilerRuntime&) = delete;
    QnnProfilerRuntime& operator=(const QnnProfilerRuntime&) = delete;

    bool Create(const QnnInterface_t* be_iface, Qnn_BackendHandle_t backend_handle, QnnProfileLevel level);

    void Destory();

    Qnn_ProfileHandle_t Handle() const {return handle_;}
    bool IsValid() const {return handle_ != nullptr;}

    Qnn_ProfileHandle_t GetProfiler(){
        return IsValid() ? Handle() : nullptr;
    }

    bool DumpEvents(bool dump_sub_events = true);

    private:
    const QnnInterface_t* be_{nullptr};
    Qnn_BackendHandle_t backend_{nullptr};
    Qnn_ProfileHandle_t handle_{nullptr};
    QnnProfileLevel level_{QnnProfileLevel::Off};

    static inline bool CheckQnnOk(Qnn_ErrorHandle_t err, const char* what){
        if(err != QNN_SUCCESS){
            std::cerr << "[QNN] " << what << " failed, err=" << QNN_GET_ERROR_CODE(err) << "\n";
            return false;
        }
        return true;
    }
    static const char* UnitSuffix(QnnProfile_EventUnit_t unit){
        switch(unit){
            case QNN_PROFILE_EVENTUNIT_MICROSEC: return " (us)";
            case QNN_PROFILE_EVENTUNIT_BYTES: return " (bytes)";
            case QNN_PROFILE_EVENTUNIT_COUNT: return " (count)";
            case QNN_PROFILE_EVENTUNIT_BACKEND:
            case QNN_PROFILE_EVENTUNIT_CYCLES:
            default: return "";
        }
    }
    static const char* UnitToStr(uint32_t unit);
    static const char* TypeToStr(uint32_t type);
};