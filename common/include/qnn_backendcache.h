#pragma once

#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>
#include <QnnTypes.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// 네가 이미 쓰는 형태에 맞춰 둠
struct QnnContextBinary {
  void* buffer{nullptr};
  uint32_t nbytes{0};
};

class QnnBackendCacheRuntime {
 public:
  enum CacheState {
    INVALID = 0,
    DESERIALIZE = 2,
    ONLINE_PREPARE = 3,
    MULTI_GRAPH = 4,
  };

  explicit QnnBackendCacheRuntime(std::string aot_graph_name = "empty_graph")
      : aot_graph_name_(std::move(aot_graph_name)) {}
  virtual ~QnnBackendCacheRuntime();

  QnnBackendCacheRuntime(const QnnBackendCacheRuntime&) = delete;
  QnnBackendCacheRuntime& operator=(const QnnBackendCacheRuntime&) = delete;

  // runtime restore 모드:
  // - sys_iface: qnn_dynload로 이미 로드된 system iface (provider[0])
  // - blob: context.bin 메모리
  bool Create(const QnnSystemInterface_t* sys_iface,
              const QnnContextBinary& blob);

  void Destroy();

  bool IsValid() const { return sys_context_handle_ != nullptr; }
  CacheState State() const { return state_; }

  std::vector<Qnn_Tensor_t> GetGraphInputs(const std::string& graph_name) const;
  std::vector<Qnn_Tensor_t> GetGraphOutputs(const std::string& graph_name) const;

 protected:
  // backend별 부가정보 파싱(HTP spillFill 등). 기본은 no-op
  virtual bool RetrieveBackendBinaryInfo(const QnnSystemContext_BinaryInfo_t* binaryinfo);

  // 내부 구현
  bool Configure();  // DESERIALIZE only
  bool GetQnnGraphInfoFromBinary(void* buffer, uint32_t nbytes);

  template <typename INFO>
  void RetrieveGraphInfo(const INFO& info);

 protected:
  const QnnSystemInterface_t* sys_{nullptr};
  QnnContextBinary blob_{};
  QnnSystemContext_Handle_t sys_context_handle_{nullptr};
  CacheState state_{INVALID};

  std::vector<std::string> graph_names_;
  std::string aot_graph_name_;

  std::unordered_map<std::string, std::vector<Qnn_Tensor_t>> input_tensor_structs_;
  std::unordered_map<std::string, std::vector<Qnn_Tensor_t>> output_tensor_structs_;
};

// HTP 전용 확장 (원하면 사용)
class HtpBackendCacheRuntime : public QnnBackendCacheRuntime {
 public:
  explicit HtpBackendCacheRuntime(std::string aot_graph_name = "graph")
      : QnnBackendCacheRuntime(std::move(aot_graph_name)) {}
  ~HtpBackendCacheRuntime() override = default;

  uint64_t GetSpillFillBufferSize() const { return spill_fill_buf_; }

 protected:
  bool RetrieveBackendBinaryInfo(const QnnSystemContext_BinaryInfo_t* binaryinfo) override;

 private:
  uint64_t spill_fill_buf_{0};
};