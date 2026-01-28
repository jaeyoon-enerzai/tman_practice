#include "qnn_tensor.h"
#include "QnnTypes.h"
#include <cstdint>

uint32_t QnnTensor::DataTypeSize(Qnn_DataType_t dt){
    switch (dt) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      return 8;
    default:
      return 0;
  }
}

uint32_t QnnTensor::CalcBytes(Qnn_DataType_t dt, const std::vector<uint32_t>& dims) {
  uint64_t n = DataTypeSize(dt);
  for (auto d : dims) n *= static_cast<uint64_t>(d);
  return static_cast<uint32_t>(n);
}

void QnnTensor::InitCommon_(Qnn_TensorType_t tensor_type, Qnn_DataType_t data_type){
    tensor_.version = QNN_TENSOR_VERSION_2;

    auto* t = QNN_TENSOR_VER_PTR(tensor_);
    t->name = name_.c_str();
    t->type = tensor_type;
    t->dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t->dataType = data_type;
    t->rank = static_cast<uint32_t>(dims_.size());
    t->dimensions = dims_.data();
    t->isDynamicDimensions = dyn_.empty() ? nullptr : dyn_.data();

    // RAW client buffer
        // clientBuf (cache도??)    
    t->memType = QNN_TENSORMEMTYPE_RAW;
    
    // TODO - quantization 추가
    t->quantizeParams = QNN_QUANTIZE_PARAMS_INIT;

    // default
    t->clientBuf.data = nullptr;
    t->clientBuf.dataSize = 0;

    if(tensor_type == QNN_TENSOR_TYPE_STATIC){
      t->clientBuf.dataSize = bytes_;
    }
}

QnnTensor::QnnTensor(std::string name,
            Qnn_TensorType_t tensor_type,
            Qnn_DataType_t data_type,
            const std::vector<uint32_t>& dims,
            const std::vector<uint8_t>* dynamic_dims,
            uint32_t bytes,
            const void* data,
            bool copy_data)
    : name_(std::move(name)), dims_(dims){
    
    if(dynamic_dims) dyn_ = *dynamic_dims;

    bytes_ = (bytes == 0) ? CalcBytes(data_type, dims_) : bytes;

    InitCommon_(tensor_type, data_type);

    if (data != nullptr){
        // static tensor
        if (tensor_type == QNN_TENSOR_TYPE_STATIC){
            (void)SetData(data, copy_data);
        }
    }
}

bool QnnTensor::AllocateAndOwn(){
    if (bytes_ == 0){
        std::cerr << "[QNN] AllocateAndOwn: bytes==0\n";
        return false;
    }
    owned_.reset(new (std::nothrow) uint8_t[bytes_]);
    if (!owned_){
        std::cerr << "[QNN] ALlocateAndOwn: OOM\n";
        return false;
    }
    auto *t = QNN_TENSOR_VER_PTR(tensor_);
    t->memType = QNN_TENSORMEMTYPE_RAW;
    t->clientBuf.dataSize = bytes_;
    t->clientBuf.data = owned_.get();
    return true;
}

bool QnnTensor::SetData(const void* data, bool copy_data){
    if (!data){
        std::cerr << "[QNN] SetData: data is null\n";
        return false;
    }
    auto* t = QNN_TENSOR_VER_PTR(tensor_);
    t->memType = QNN_TENSORMEMTYPE_RAW;
    t->clientBuf.dataSize = bytes_;

    if(copy_data){
        if(!owned_ && !AllocateAndOwn()) return false;
        std::memcpy(owned_.get(), data, bytes_);
        t->clientBuf.data = owned_.get();
    } else{
        t->clientBuf.data = const_cast<void*>(data);
    }
    return true;
}

bool QnnTensor::SetDims(const std::vector<uint32_t>& dims) {
  if (dims.size() != dims_.size()) {
    std::cerr << "[QNN] SetDims: rank mismatch\n";
    return false;
  }
  dims_ = dims;
  QNN_TENSOR_VER_PTR(tensor_)->dimensions = dims_.data();
  return true;
}

void QnnTensor::UpdateMetaFrom(const Qnn_Tensor_t& created_tensor) {
  // 최소로 id만 복사 (execuTorch도 이 정도만 했음)
  QNN_TENSOR_VER_PTR(tensor_)->id = QNN_TENSOR_VER_PTR(created_tensor)->id;
}

bool QnnTensor::SetName(const std::string& new_name){
    name_ = new_name;
    QNN_TENSOR_VER_PTR(tensor_)->name = name_.c_str();
    return true;
}