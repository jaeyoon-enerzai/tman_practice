#pragma once
#include <array>
#include <cstdint>
#include <iostream>
#include <list>
#include <fstream>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "QnnProfile.h"
#include "QnnCommon.h"
#include "System/QnnSystemProfile.h"

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

    bool Create(const QnnInterface_t* be_iface, const QnnSystemInterface_t* sys_iface, Qnn_BackendHandle_t backend_handle, QnnProfileLevel level,
                bool enable_serialization, const std::string& log_filename);

    void Destory();

    Qnn_ProfileHandle_t Handle() const {return handle_;}

    bool SerializeAfterExecute(const char* graphName);

    bool IsValid() const {return handle_ != nullptr;}

    Qnn_ProfileHandle_t GetProfiler(){
        return IsValid() ? Handle() : nullptr;
    }

    bool DumpEventsRecursive(bool dump_sub_events, int max_depth);

    private:
    bool TryDumpOpaqueObjectToFile(const QnnProfile_ExtendedEventDataV1_t& v,
                                 const std::string& tag,
                                 int depth);
    bool DumpOneEventRecursive(QnnProfile_EventId_t eventId,
                                              int depth,
                                              int max_depth);
    static uint64_t NowUs();
    const QnnInterface_t* be_{nullptr};
    const QnnSystemInterface_t* sys_{nullptr};
    Qnn_BackendHandle_t backend_{nullptr};
    Qnn_ProfileHandle_t handle_{nullptr};
    QnnProfileLevel level_{QnnProfileLevel::Off};
    bool enable_serialization_{false};
    std::string log_filename_;
    QnnSystemProfile_SerializationTargetHandle_t ser_target_{nullptr};

    bool CreateSerializationTarget();
    void FreeSerializationTarget();
    bool ExtractBackendProfilingInfo(QnnSystemProfile_ProfileData_t* profileData);
    bool ExtractProfileingEvent(QnnProfile_EventId_t profileEventId, QnnSystemProfile_ProfileEventV1_t& out);
    bool ExtractProfilingSubEvents(QnnProfile_EventId_t profileEventId, QnnSystemProfile_ProfileEventV1_t& parent,
                        std::list<std::vector<QnnSystemProfile_ProfileEventV1_t>>& arena);

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