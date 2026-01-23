#pragma once
#include <memory>
#include <vector>
#include <cstring>
#include <iostream>

#include "HTP/QnnHtpGraph.h"
#include "QnnGraph.h"
#include "qnn_option.h"

class QnnHtpGraphCustomConfigRuntime{
    public:
        QnnHtpGraphCustomConfigRuntime(uint32_t vtcm_mb,
                                    float opt_level,
                                    bool enable_dlbc)
            :vtcm_mb_(vtcm_mb), opt_level_(opt_level), enable_dlbc_(enable_dlbc){}
        std::vector<QnnGraph_CustomConfig_t> Create();

    private:
        QnnHtpGraph_CustomConfig_t* Alloc(){
            cfg_.emplace_back(std::make_unique<QnnHtpGraph_CustomConfig_t>());
            std::memset(cfg_.back().get(), 0, sizeof(QnnHtpGraph_CustomConfig_t));
            cfg_.back()->option = QNN_HTP_GRAPH_CONFIG_OPTION_UNKNOWN;
            return cfg_.back().get();
        }

        uint32_t vtcm_mb_;
        float opt_level_;
        bool enable_dlbc_;
        std::vector<std::unique_ptr<QnnHtpGraph_CustomConfig_t>> cfg_;
};