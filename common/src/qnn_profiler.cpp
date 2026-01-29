#include "qnn_profiler.h"
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnProfile.h"
#include "HTP/QnnHtpProfile.h"
#include <cstddef>
#include <cstdint>

bool QnnProfilerRuntime::Create(const QnnInterface_t* be_iface,
                            Qnn_BackendHandle_t backend_handle,
                            QnnProfileLevel level
){
    if (!be_iface || !backend_handle){
        std::cerr << "[QNN] Profiler Create: invalid be/backend\n";
        return false;
    }
    be_ = be_iface;
    backend_ = backend_handle;
    level_ = level;

    if(level == QnnProfileLevel::Off){
        handle_ = nullptr;
        return true;
    }
    if(handle_) return true; // already created

    auto & api = be_->QNN_INTERFACE_VER_NAME;

    QnnProfile_Level_t qnn_level;
    if(level == QnnProfileLevel::Basic){
        qnn_level = QNN_PROFILE_LEVEL_BASIC;
    } else if (level == QnnProfileLevel::Detailed || level == QnnProfileLevel::Optrace){
        // qnn_level = QNN_PROFILE_LEVEL_DETAILED;
        qnn_level = QNN_HTP_PROFILE_LEVEL_LINTING;
    } else{
        std::cerr << "[QNN] Profiler Create : invalid qnn_level\n";
        return false;
    }
    std::cout << "==========xxx=========xxxxxxxxxxx====" << std::endl;
    std::cout << "QNN LEVEL IS " << qnn_level << std::endl;

    Qnn_ProfileHandle_t ph = nullptr;
    if(!CheckQnnOk(api.profileCreate(backend_, qnn_level, &ph), "profileCreate")){
        handle_ =nullptr;
        return false;
    }
    handle_ = ph;

    if (level == QnnProfileLevel::Optrace){
        if(!handle_){
            std::cerr << "[QNN] Profiler: null handle, cannot enable optrace\n";
            return false;
        }
        QnnProfile_Config_t cfg = QNN_PROFILE_CONFIG_INIT;
        cfg.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;

        QnnProfile_Config_t cfg_max = QNN_PROFILE_CONFIG_INIT;
        cfg_max.option = QNN_PROFILE_CONFIG_OPTION_MAX_EVENTS;
        cfg_max.numMaxEvents = 300;
        std::array<const QnnProfile_Config_t*, 3> cfgs = {&cfg, &cfg_max, nullptr};
        
        if(!CheckQnnOk(api.profileSetConfig(handle_, cfgs.data()), "profileSetConfig(ENABLE_OPTRACE)")){
            std::cout << "Warning - Failed to set optrace for backend\n";
        }
    }
    return true;
}

void QnnProfilerRuntime::Destory(){
    if(!be_ || !handle_){
        handle_ = nullptr;
        return;
    }
    auto& api = be_->QNN_INTERFACE_VER_NAME;
    (void)CheckQnnOk(api.profileFree(handle_), "profileFree");
    handle_ = nullptr;
}

const char* QnnProfilerRuntime::UnitToStr(uint32_t unit) {
  switch (unit) {
    case QNN_PROFILE_EVENTUNIT_MICROSEC: return "us";
    case QNN_PROFILE_EVENTUNIT_BYTES:    return "bytes";
    case QNN_PROFILE_EVENTUNIT_CYCLES:   return "cycles";
    case QNN_PROFILE_EVENTUNIT_COUNT:    return "count";
    case QNN_PROFILE_EVENTUNIT_OBJECT:   return "object";
    case QNN_PROFILE_EVENTUNIT_NONE:     return "none";
    default: return "unit?";
  }
}

const char* QnnProfilerRuntime::TypeToStr(uint32_t type) {
  switch (type) {
    case QNN_PROFILE_EVENTTYPE_INIT:    return "INIT";
    case QNN_PROFILE_EVENTTYPE_FINALIZE:return "FINALIZE";
    case QNN_PROFILE_EVENTTYPE_EXECUTE: return "EXECUTE";
    case QNN_PROFILE_EVENTTYPE_NODE:    return "NODE";
    case QNN_PROFILE_EVENTTYPE_TRACE:   return "TRACE";
    case QNN_PROFILE_EVENTTYPE_DEINIT:  return "DEINIT";
    default: return "TYPE?";
  }
}

bool QnnProfilerRuntime::DumpEvents(bool dump_sub_events){
    if(!be_ || !handle_) return false;
    auto& api = be_->QNN_INTERFACE_VER_NAME;

    const QnnProfile_EventId_t* events = nullptr;
    uint32_t num_events = 0;

    if(!CheckQnnOk(api.profileGetEvents(handle_, &events, &num_events), "profileGetEvents")){
        return false;
    }
    std::cout << "[QNN] Profile events : " << num_events << "\n";

    for(uint32_t i=0; i < num_events; ++i){
        QnnProfile_EventData_t ed = QNN_PROFILE_EVENT_DATA_INIT;
        Qnn_ErrorHandle_t err_ed = api.profileGetEventData(events[i], &ed);

        if(err_ed == QNN_SUCCESS){
            const char* ident = ed.identifier ? ed.identifier : "(null)";
            std::cout << "[E] type=" << TypeToStr(ed.type)
                << "(" << ed.type << ") "
                << "unit=" << UnitToStr(ed.unit) << "(" << ed.unit << ") "
                << "value=" << (unsigned long long)ed.value
                << " ident=" << ident << "\n";
        } else{
            QnnProfile_ExtendedEventData_t x = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
            Qnn_ErrorHandle_t err_x = api.profileGetExtendedEventData(events[i], &x);
            if (!CheckQnnOk(err_x, "profileGetExtendedEventData")) return false;

            auto& v = x.v1;
            const char* ident = v.identifier ? v.identifier : "(null)";

            std::cout << "[X] type=" << TypeToStr(v.type)
                        << "(" << v.type << ") "
                        << "unit=" << UnitToStr(v.unit) << "(" << v.unit << ") "
                        << "ts(us)=" << (unsigned long long)v.timestamp
                        << " ident=" << ident;

            if (v.unit == QNN_PROFILE_EVENTUNIT_OBJECT) {
                const char* fn = v.backendOpaqueObject.fileName ? v.backendOpaqueObject.fileName : "(null)";
                std::cout << " OBJECT.fileName=" << fn;
                if (v.backendOpaqueObject.fileName) {
                std::cout << "\n      >>> CANDIDATE RAW LOG FILE (for qnn-profiler-viewer): " << fn;
                }
            } else {
                // scalar value
                // v.value is Qnn_Scalar_t (union-like), safest is print common fields
                // QNN_SCALAR_INIT uses 'dataType' + union. But not guaranteed what backend uses.
                // We'll print raw 64 if possible by reading as uint64.
                std::cout << " scalar";
            }
            std::cout << "\n";
        }

        if(!dump_sub_events) continue;

        // sub events (노드 타임 등)
        const QnnProfile_EventId_t* sub = nullptr;
        uint32_t num_sub = 0;
        Qnn_ErrorHandle_t err = api.profileGetSubEvents(events[i], &sub, &num_sub);
        if (err != QNN_SUCCESS) {
            // 모든 이벤트가 subevent를 가지는 건 아니라서, 실패를 fatal로 보지 않게 처리 가능
            continue;
        }

        for (uint32_t j = 0; j < num_sub; ++j) {
            QnnProfile_EventData_t sd{};
            if (!CheckQnnOk(api.profileGetEventData(sub[j], &sd), "profileGetEventData(sub)")) return false;

            // ExecuTorch처럼 NODE + (us/cycles)만 출력하고 싶으면 여기서 필터링 가능
            std::cout << "      * " << (sd.identifier ? sd.identifier : "<null>")
                        << UnitSuffix(sd.unit)
                        << " : " << sd.value
                        << " (type=" << sd.type << ")\n";
            
            QnnProfile_ExtendedEventData_t sx = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
            Qnn_ErrorHandle_t err_sx = api.profileGetExtendedEventData(sub[j], &sx);
            if(err_sx == QNN_SUCCESS){
                auto& v = sx.v1;
                const char* ident = v.identifier ? v.identifier : "(null)";
                std::cout << "    [SX] type=" << TypeToStr(v.type)
                        << "(" << v.type << ") "
                        << "unit=" << UnitToStr(v.unit) << "(" << v.unit << ") "
                        << "ts(us)=" << (unsigned long long)v.timestamp
                        << " ident=" << ident;
                if (v.unit == QNN_PROFILE_EVENTUNIT_OBJECT) {
                const char* fn = v.backendOpaqueObject.fileName ? v.backendOpaqueObject.fileName : "(null)";
                std::cout << " OBJECT.fileName=" << fn;
                if (v.backendOpaqueObject.fileName) {
                    std::cout << "\n        >>> CANDIDATE RAW LOG FILE: " << fn;
                }
                }
                std::cout << "\n";
            }
        }
    }
    return true;
}