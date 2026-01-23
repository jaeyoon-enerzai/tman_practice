#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>

#include "QnnTypes.h"
#include "QnnCommon.h"

#ifndef QNN_TENSOR_VER_PTR
#define QNN_TENSOR_VER_PTR(x) (&((x).v2))
#endif

class QnnTensor{
    public:
    QnnTensor() = default;

    // dynamic_dims : rank 길이만큼의 0,1로 이루어진 배열 (nullptr if none)
    // bytes ==0 => dims * dtypes_size
    QnnTensor(std::string name,
            Qnn_TensorType_t tensor_type,
            Qnn_DataType_t data_type,
            const std::vector<uint32_t>& dims,
            const std::vector<uint8_t>* dynamic_dims = nullptr,
            uint32_t bytes = 0,
            const void* data = nullptr,
            bool copy_data = false);
    const std::string& Name() const {return name_;}
    uint32_t Rank() const { return static_cast<uint32_t>(dims_.size());}
    
    Qnn_Tensor_t Clone() const { return tensor_;}

    // static tensor
    bool AllocateAndOwn();
    bool SetData(const void* data, bool copy_data=false);

    // when shape changes during runtime
    bool SetDims(const std::vector<uint32_t>& dims);

    // graph tensor create 이후 id 같은 메타 업데이트 시 반영
    void UpdateMetaFrom(const Qnn_Tensor_t& created_tensor);

    Qnn_Tensor_t* MutableTensor() {return &tensor_;}
    const Qnn_Tensor_t* Tensor() const {return &tensor_;}

    bool IsCreated() const { return created_;}
    void MarkCreated() { created_ = true; }

    static uint32_t DataTypeSize(Qnn_DataType_t dt);
    static uint32_t CalcBytes(Qnn_DataType_t dt, const std::vector<uint32_t>& dims);
    bool SetName(const std::string& new_name);

    private:
    void InitCommon_(Qnn_TensorType_t tensor_type, Qnn_DataType_t data_type);

    std::string name_;
    std::vector<uint32_t> dims_;
    std::vector<uint8_t> dyn_;
    uint32_t bytes_{0};
    bool created_{false};

    std::unique_ptr<uint8_t[]> owned_;

    Qnn_Tensor_t tensor_{.version = QNN_TENSOR_VERSION_2, .v2= QNN_TENSOR_V2_INIT};
};
