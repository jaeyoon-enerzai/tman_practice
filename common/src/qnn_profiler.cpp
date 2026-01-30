#include "qnn_profiler.h"
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnProfile.h"
#include "HTP/QnnHtpProfile.h"
#include "System/QnnSystemProfile.h"
#include <cstddef>
#include <cstdint>
#include <vector>

uint64_t QnnProfilerRuntime::NowUs() {
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}


bool QnnProfilerRuntime::Create(const QnnInterface_t* be_iface,
                            const QnnSystemInterface_t* sys_iface,
                            Qnn_BackendHandle_t backend_handle,
                            QnnProfileLevel level, bool enable_serialization,
                            const std::string& log_filename
){
    if (!be_iface || !backend_handle || !sys_iface){
        std::cerr << "[QNN] Profiler Create: invalid be/backend\n";
        return false;
    }
    be_ = be_iface;
    backend_ = backend_handle;
    sys_ = sys_iface;
    level_ = level;
    enable_serialization_ = enable_serialization;
    log_filename_ = log_filename;

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
        qnn_level = QNN_PROFILE_LEVEL_DETAILED;
        // qnn_level = QNN_HTP_PROFILE_LEVEL_LINTING;
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
        cfg.enableOptrace = true;

        QnnProfile_Config_t cfg_max = QNN_PROFILE_CONFIG_INIT;
        cfg_max.option = QNN_PROFILE_CONFIG_OPTION_MAX_EVENTS;
        cfg_max.numMaxEvents = 300;
        // std::array<const QnnProfile_Config_t*, 3> cfgs = {&cfg, &cfg_max, nullptr};
        std::array<const QnnProfile_Config_t*, 2> cfgs = {&cfg, nullptr};
        
        std::cout << "OPTRACE SET\n";
        
        auto errCfg = api.profileSetConfig(handle_, cfgs.data());
        if (errCfg != QNN_SUCCESS) {
            std::cerr << "profileSetConfig failed, err=" << QNN_GET_ERROR_CODE(errCfg) << "\n";
            std::cout << "profileSetConfig failed, err=" << QNN_GET_ERROR_CODE(errCfg) << "\n";
            // return false;
        } else{
            std::cout << "PROFILER CONFIG SET\n";
        }
        // if(!CheckQnnOk(api.profileSetConfig(handle_, cfgs.data()), "profileSetConfig(ENABLE_OPTRACE)")){
        //     std::cout << "Warning - Failed to set optrace for backend\n";
        // }
    }

    if(enable_serialization_){
        if(!CreateSerializationTarget()){
            std::cerr << "[QNN] Failed to create serialization target\n";
            Destory();
            return false;
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
    sys_ = nullptr;
    backend_ = nullptr;
    be_ = nullptr;

    FreeSerializationTarget();
}

bool QnnProfilerRuntime::CreateSerializationTarget(){
    if (!sys_) return false;

    auto& sysapi = sys_->QNN_SYSTEM_INTERFACE_VER_NAME;

    if(!sysapi.systemProfileCreateSerializationTarget ||
        !sysapi.systemProfileSerializeEventData ||
        !sysapi.systemProfileFreeSerializationTarget){
        std::cerr << "[QNN] System profile serialization APIs are missing (nullptr)\n";
        return false;
    }
#if defined(__aarch64__)
    {
      std::string full = "/data/local/tmp/htprun";
      if (!full.empty() && full.back() != '/') full.push_back('/');
      full += log_filename_;
      std::ofstream trunc(full, std::ios::binary | std::ios::trunc);
      if (!trunc.good()) {
        std::cerr << "[QNN] Failed to truncate log file: " << full << "\n";
        return false;
      }
    }
#endif

  const char* backendBuildIdC = nullptr;
  if (be_ && be_->QNN_INTERFACE_VER_NAME.backendGetBuildId) {
    (void)be_->QNN_INTERFACE_VER_NAME.backendGetBuildId(&backendBuildIdC);
  }
  std::string backendBuildId = backendBuildIdC ? backendBuildIdC : "";

  QnnSystemProfile_SerializationFileHeader_t header{};
  header.appName        = "tmanprac";
  header.appVersion     = "1.0";
  header.backendVersion = backendBuildId.c_str();

  QnnSystemProfile_SerializationTargetFile_t file{};
  file.fileName      = log_filename_.c_str();
  file.fileDirectory = "/data/local/tmp/htprun";

  QnnSystemProfile_SerializationTarget_t target{};
  target.type = QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE;
  target.file = file;

  QnnSystemProfile_SerializationTargetConfig_t cfg{};
  cfg.type = QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER;
  cfg.serializationHeader = header;

  QnnSystemProfile_SerializationTargetHandle_t handle = nullptr;
  Qnn_ErrorHandle_t err =
      sysapi.systemProfileCreateSerializationTarget(target, &cfg, 1, &handle);
  if (!CheckQnnOk(err, "systemProfileCreateSerializationTarget")) {
    return false;
  }

  ser_target_ = handle;
  return true;
}

void QnnProfilerRuntime::FreeSerializationTarget(){
    if (!sys_ || !ser_target_) return;
    
    auto& sysapi = sys_->QNN_SYSTEM_INTERFACE_VER_NAME;
    if (sysapi.systemProfileFreeSerializationTarget) {
        (void)CheckQnnOk(sysapi.systemProfileFreeSerializationTarget(ser_target_),
                        "systemProfileFreeSerializationTarget");
    }
    ser_target_ = nullptr;
}

bool QnnProfilerRuntime::SerializeAfterExecute(const char* graphName) {
  if (!enable_serialization_ || !ser_target_) return true; // serialization 안 쓰면 no-op
  if (!handle_) return false;

  QnnSystemProfile_ProfileData_t pd = QNN_SYSTEM_PROFILE_DATA_INIT;
  pd.version = QNN_SYSTEM_PROFILE_DATA_VERSION_1;
  pd.v1.header.methodType = QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE;
  pd.v1.header.startTime  = NowUs(); // 정확히 execute 전/후 타임스탬프를 잡고싶으면 main에서 넘겨도 됨
  pd.v1.header.stopTime   = NowUs();
  pd.v1.header.graphName  = graphName; // nullptr 가능

  if (!ExtractBackendProfilingInfo(&pd)) {
    std::cerr << "[QNN] ExtractBackendProfilingInfo failed\n";
    return false;
  }
  return true;
}

bool QnnProfilerRuntime::ExtractBackendProfilingInfo(QnnSystemProfile_ProfileData_t* profileData){
    if (!be_ || !sys_ || !ser_target_ || !handle_ || !profileData) return false;

    auto& api = be_->QNN_INTERFACE_VER_NAME;
    auto& sysapi = sys_->QNN_SYSTEM_INTERFACE_VER_NAME;

    const QnnProfile_EventId_t* events = nullptr;
    uint32_t numEvents = 0;
    if(!CheckQnnOk(api.profileGetEvents(handle_, &events, &numEvents), "profileGetEvnets")){
        return false;
    }

    std::vector<QnnSystemProfile_ProfileEventV1_t> top;
    std::list<std::vector<QnnSystemProfile_ProfileEventV1_t>> arena; // keep subvector lifetime

    top.reserve(numEvents);

    for(uint32_t i=0; i< numEvents; ++i){
        QnnSystemProfile_ProfileEventV1_t e{};
        if(!ExtractProfileingEvent(events[i], e)) return false;

        if(!ExtractProfilingSubEvents(events[i], e, arena)) return false;
        top.push_back(e);
    }

    profileData->v1.profilingEvents = top.data();
    profileData->v1.numProfilingEvents = static_cast<uint32_t>(top.size());

    const QnnSystemProfile_ProfileData_t* arr[1] = {profileData};
    Qnn_ErrorHandle_t err = sysapi.systemProfileSerializeEventData(ser_target_, arr, 1);
    if (!CheckQnnOk(err, "systemProfileSerializeEventData")) return false;

    return true;
}

bool QnnProfilerRuntime::ExtractProfileingEvent(QnnProfile_EventId_t profileEventId, 
                                        QnnSystemProfile_ProfileEventV1_t& out ){
    if (!be_) return false;
    auto& api = be_->QNN_INTERFACE_VER_NAME;

    QnnProfile_EventData_t ed = QNN_PROFILE_EVENT_DATA_INIT;
    if(!CheckQnnOk(api.profileGetEventData(profileEventId, &ed), "profileGetEventData")) return false;

    if(ed.unit != QNN_PROFILE_EVENTUNIT_OBJECT){
        out.type = QNN_SYSTEM_PROFILE_EVENT_DATA;
        out.eventData = ed;
        out.profileSubEventData = nullptr;
        out.numSubEvents = 0;
        return true;
    } else{
        QnnProfile_ExtendedEventData_t xd = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
        if(!CheckQnnOk(api.profileGetExtendedEventData(profileEventId, &xd), "profileGetExtendedEventData")) return false;

        out.type = QNN_SYSTEM_PROFILE_EXTENDED_EVENT_DATA;
        out.extendedEventData = xd;
        out.profileSubEventData = nullptr;
        out.numSubEvents = 0;
        return true;
    }
}

bool QnnProfilerRuntime::ExtractProfilingSubEvents(QnnProfile_EventId_t profileEventId,
    QnnSystemProfile_ProfileEventV1_t& parent,
    std::list<std::vector<QnnSystemProfile_ProfileEventV1_t>>& arena){
    if (!be_) return false;
    auto& api = be_->QNN_INTERFACE_VER_NAME;

    const QnnProfile_EventId_t* sub = nullptr;
    uint32_t numSub = 0;

    Qnn_ErrorHandle_t err = api.profileGetSubEvents(profileEventId, &sub, &numSub);
    if (err != QNN_SUCCESS){
        parent.profileSubEventData = nullptr;
        parent.numSubEvents = 0;
        return true;
    }

    std::vector<QnnSystemProfile_ProfileEventV1_t> vec;
    vec.reserve(numSub);

    for(uint32_t i = 0; i< numSub; ++i){
        QnnSystemProfile_ProfileEventV1_t child{};
        if(!ExtractProfileingEvent(sub[i], child)) return false;

        // recursive
        if(!ExtractProfilingSubEvents(sub[i], child, arena)) return false;
        vec.push_back(child);
    }

    arena.push_back(std::move(vec));
    parent.profileSubEventData = arena.back().data();
    parent.numSubEvents = static_cast<uint32_t>(arena.back().size());
    return true;
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
    default: break;
  }

  // HTP 확장(네가 보고 있는 3001, 3002 등)
  switch (type) {
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_RPC_TIME_MICROSEC:      return "HTP_EXEC_HOST_RPC_US";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HTP_RPC_TIME_MICROSEC:       return "HTP_EXEC_HTP_RPC_US";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE:            return "HTP_EXEC_ACCEL_CYCLES";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC:         return "HTP_EXEC_ACCEL_US";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_EXCL_WAIT_TIME_MICROSEC:return "HTP_EXEC_ACCEL_EXCL_WAIT_US";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_YIELD_COUNT:                 return "HTP_YIELD_COUNT";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_VTCM_ACQUIRE_TIME:           return "HTP_VTCM_ACQUIRE_US";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_RESOURCE_POWER_UP_TIME:      return "HTP_POWERUP_US";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_CRITICAL_ACCEL_TIME_CYCLE:   return "HTP_CRITICAL_CYCLES";
    case QNN_HTP_PROFILE_EVENTTYPE_GRAPH_NUMBER_OF_HVX_THREADS:               return "HTP_HVX_THREADS";

    // linting 관련(원하면 나중에 optrace와 별개로 수집)
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT:         return "HTP_NODE_WAIT";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_OVERLAP:      return "HTP_NODE_OVERLAP";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT_OVERLAP: return "HTP_NODE_WAIT_OVERLAP";
    case QNN_HTP_PROFILE_EVENTTYPE_NODE_RESOURCEMASK: return "HTP_NODE_RES_MASK";
    default: break;
  }
  return "TYPE?";
}

bool QnnProfilerRuntime::TryDumpOpaqueObjectToFile(const decltype(QnnProfile_ExtendedEventData_t{}.v1)& v,
                                                   const std::string& tag,
                                                   int depth) {
  // fileName이 이미 있으면 굳이 안 덤프해도 됨
  const char* fn = v.backendOpaqueObject.fileName;
  std::cout << "DUMP!!!!\n";
  if (fn && fn[0] != '\0') return true;

  // payload가 없으면 덤프 불가
  const void* p = v.backendOpaqueObject.opaqueObject.data;
  uint64_t sz   = (uint64_t)v.backendOpaqueObject.opaqueObject.len;
  if (!p || sz == 0) return true;

  // 파일명 생성
  std::string dir = "/data/local/tmp/htprun";
  std::string out = dir + "/qnn-optrace-" + std::to_string((unsigned long long)NowUs())
                  + "-d" + std::to_string(depth) + "-" + tag + ".bin";

  std::ofstream ofs(out, std::ios::binary);
  if (!ofs.good()) {
    std::cerr << "[QNN] Failed to open dump file: " << out << "\n";
    return false;
  }
  ofs.write(reinterpret_cast<const char*>(p), (std::streamsize)sz);
  ofs.close();

  std::cout << "\n      >>> DUMPED TRACE PAYLOAD to: " << out
            << " (bytes=" << (unsigned long long)sz << ")\n";
  return true;
}

bool QnnProfilerRuntime::DumpEventsRecursive(bool dump_sub_events, int max_depth) {
  if (!be_ || !handle_) return false;
  auto& api = be_->QNN_INTERFACE_VER_NAME;

  const QnnProfile_EventId_t* events = nullptr;
  uint32_t num_events = 0;

  if (!CheckQnnOk(api.profileGetEvents(handle_, &events, &num_events), "profileGetEvents")) {
    return false;
  }

  std::cout << "[QNN] Profile events: " << num_events << "\n";
  for (uint32_t i = 0; i < num_events; ++i) {
    if (!DumpOneEventRecursive(events[i], /*depth=*/0, max_depth)) return false;
    if (!dump_sub_events) continue;
  }
  return true;
}

bool QnnProfilerRuntime::DumpOneEventRecursive(QnnProfile_EventId_t eventId,
                                              int depth,
                                              int max_depth) {
  if (!be_ || !handle_) return false;
  auto& api = be_->QNN_INTERFACE_VER_NAME;

  auto indent = [&](int d) {
    for (int k = 0; k < d; ++k) std::cout << "  ";
  };

  // 1) try basic event data
  QnnProfile_EventData_t ed = QNN_PROFILE_EVENT_DATA_INIT;
  Qnn_ErrorHandle_t err_ed = api.profileGetEventData(eventId, &ed);

  if (err_ed == QNN_SUCCESS) {
    const char* ident = ed.identifier ? ed.identifier : "(null)";
    indent(depth);
    std::cout << "[E] type=" << TypeToStr(ed.type) << "(" << ed.type << ") "
              << "unit=" << UnitToStr(ed.unit) << "(" << ed.unit << ") "
              << "value=" << (unsigned long long)ed.value
              << " ident=" << ident << "\n";
    if(ed.type == QNN_PROFILE_EVENTTYPE_TRACE && ed.unit == QNN_PROFILE_EVENTUNIT_OBJECT){
        QnnProfile_ExtendedEventData_t x = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
        if (api.profileGetExtendedEventData(eventId, &x) == QNN_SUCCESS){
            auto&v  = x.v1;
            const char* ident2 = v.identifier ? v.identifier : "(null)";
            indent(depth);
            std::cout << "    [TRACE-X] type=" << TypeToStr(v.type) << "(" << v.type << ") "
                        << "unit=" << UnitToStr(v.unit) << "(" << v.unit << ") "
                        << "ts(us)=" << (unsigned long long)v.timestamp
                        << " ident=" << ident2;

            if (v.unit == QNN_PROFILE_EVENTUNIT_OBJECT) {
                const char* fn = v.backendOpaqueObject.fileName ? v.backendOpaqueObject.fileName : "(null)";
                std::cout << " OBJECT.fileName=" << fn << " OBJECT.opaqueObject=<present> " << " size : " << (unsigned long long)v.backendOpaqueObject.opaqueObject.len;
                std::string tag = (v.type == QNN_PROFILE_EVENTTYPE_TRACE) ? "TRACE" : "OBJ";
                if (!TryDumpOpaqueObjectToFile(v, tag, depth)) return false;
            }
            std::cout << "\n";
        }
    }
  } else {
    // 2) extended
    QnnProfile_ExtendedEventData_t x = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
    if (!CheckQnnOk(api.profileGetExtendedEventData(eventId, &x), "profileGetExtendedEventData")) {
      return false;
    }
    auto& v = x.v1;
    const char* ident = v.identifier ? v.identifier : "(null)";

    indent(depth);
    std::cout << "[X] type=" << TypeToStr(v.type) << "(" << v.type << ") "
              << "unit=" << UnitToStr(v.unit) << "(" << v.unit << ") "
              << "ts(us)=" << (unsigned long long)v.timestamp
              << " ident=" << ident;

    if (v.unit == QNN_PROFILE_EVENTUNIT_OBJECT) {
      const char* fn = v.backendOpaqueObject.fileName ? v.backendOpaqueObject.fileName : "(null)";
      std::cout << " OBJECT.fileName=" << fn;
      // opaqueObject는 캐스팅하지 말 것 (너가 본 컴파일 에러 원인)
      std::cout << " OBJECT.opaqueObject=<present>";

      std::string tag = (v.type == QNN_PROFILE_EVENTTYPE_TRACE)? "TRACE" : "OBJ";
      std::cout << "GET TAG " << tag.c_str() << std::endl;
      if(!TryDumpOpaqueObjectToFile(v, tag, depth)) return false;
    } else {
      std::cout << " scalar";
    }
    std::cout << "\n";
  }

  // stop recursion
  if (depth >= max_depth) return true;

  // 3) sub events
  const QnnProfile_EventId_t* sub = nullptr;
  uint32_t num_sub = 0;
  Qnn_ErrorHandle_t err_sub = api.profileGetSubEvents(eventId, &sub, &num_sub);
  if (err_sub != QNN_SUCCESS) {
    return true; // sub가 없는 이벤트도 많음
  }

  indent(depth);
  std::cout << "  -> subEvents=" << num_sub << "\n";

  for (uint32_t j = 0; j < num_sub; ++j) {
    if (!DumpOneEventRecursive(sub[j], depth + 1, max_depth)) return false;
  }
  return true;
}