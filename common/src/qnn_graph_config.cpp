#include "qnn_graph_config.h"
#include "HTP/QnnHtpGraph.h"
#include "QnnGraph.h"
#include "QnnTypes.h"
#include <vector>

std::vector<QnnGraph_CustomConfig_t> QnnHtpGraphCustomConfigRuntime::Create(){
    cfg_.clear();

    std::vector<QnnGraph_CustomConfig_t> ret;
    QnnHtpGraph_CustomConfig_t* p = nullptr;

    // 안해도 된다는데 일단 코드만 넣어놓는다
    // (A) precision: 지금은 FP32로 고정 (FP16 금지)
    // if (precision_ == QnnHtpGraphPrecision::kFp16) {
        // p = Alloc();
        // p->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        // p->precision = QNN_PRECISION_FLOAT16;
        // ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p));
        // default/quantized면 굳이 안 넣어도 됨


    // p = Alloc();
    // p->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    // p->precision = QNN_PRECISION_FLOAT32;
    // ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p));

    // 굳이 안할 이유가 없어 보임
    p = Alloc();
    p->option = QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF;
    p->shortDepthConvOnHmxOff = true;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p));

    p = Alloc();
    p->option = QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
    p->foldReluActivationIntoConvOff = true;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p));

    p = Alloc();
    p->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    p->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    p->optimizationOption.floatValue = opt_level_;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p));

    p = Alloc();
    p->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
    p->vtcmSizeInMB = vtcm_mb_;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p));

    p = Alloc();
    p->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    p->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
    p->optimizationOption.floatValue = enable_dlbc_? 1.0f : 0.0f;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p));

    return ret;
}