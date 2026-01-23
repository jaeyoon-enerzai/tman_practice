#pragma once
#include <cstdio>
#include <cstdarg>
#include <chrono>
#include "QnnLog.h"
#include "QnnInterface.h"

static double now_ms() {
  using namespace std::chrono;
  static const auto t0 = steady_clock::now();
  const auto t  = steady_clock::now();
  return duration<double, std::milli>(t - t0).count();
}

static const char* levelToStr(QnnLog_Level_t level) {
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:   return "ERROR";
    case QNN_LOG_LEVEL_WARN:    return "WARN";
    case QNN_LOG_LEVEL_INFO:    return "INFO";
    case QNN_LOG_LEVEL_DEBUG:   return "DEBUG";
    case QNN_LOG_LEVEL_VERBOSE: return "VERBOSE";
    case QNN_LOG_LEVEL_MAX:     return "UNKNOWN";
    default:                    return "UNKNOWN";
  }
}

static void logStdoutCallback(const char* fmt,
                              QnnLog_Level_t level,
                              uint64_t timestamp,
                              va_list argp) {
  // timestamp는 backend가 주는 값인데 포맷/단위가 backend마다 다를 수 있어서
  // 우선 host 기준 ms도 같이 찍어두는게 디버깅에 좋아.
  (void)timestamp;

  std::fprintf(stdout, "%8.1fms [%-7s] ", now_ms(), levelToStr(level));
  std::vfprintf(stdout, fmt, argp);
  std::fprintf(stdout, "\n");
  std::fflush(stdout);
}

static bool CreateQnnLogger(const QnnInterface_t* be_iface,
                            Qnn_LogHandle_t* out_log,
                            QnnLog_Level_t max_level = QNN_LOG_LEVEL_VERBOSE) {
  if (!be_iface || !out_log) return false;
  *out_log = nullptr;

  auto& api = be_iface->QNN_INTERFACE_VER_NAME;
  if (!api.logCreate) {
    std::fprintf(stderr, "[QNN] logCreate not available in this interface\n");
    return false;
  }

  const Qnn_ErrorHandle_t err = api.logCreate(logStdoutCallback, max_level, out_log);
  if (err != QNN_SUCCESS || !*out_log) {
    std::fprintf(stderr, "[QNN] logCreate failed, err=%d\n", QNN_GET_ERROR_CODE(err));
    return false;
  }
  return true;
}

static void FreeQnnLogger(const QnnInterface_t* be_iface, Qnn_LogHandle_t* log) {
  if (!be_iface || !log || !*log) return;
  auto& api = be_iface->QNN_INTERFACE_VER_NAME;
  if (api.logFree) {
    api.logFree(*log);
  }
  *log = nullptr;
}