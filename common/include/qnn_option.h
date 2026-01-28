#pragma once
#include <cstdint>

enum class QnnHtpPdSession : int32_t {
  kHtpUnsignedPd = 0,
  kHtpSignedPd = 1,
  MIN = kHtpUnsignedPd,
  MAX = kHtpSignedPd
};

enum class QcomChipset: int{
    UNKNOWN_SM = 0,
    SA8295 = 39,
    SM8450 = 36,
    SM8475 = 42,
    SM8550 = 43,
    SM8650 = 57,
    SM8750 = 69,
    SSG2115P = 46,
    SSG2125P = 58,
    SXR1230P = 45,
    SXR2230P = 53,
    SXR2330P = 75,
    SM8850=87,
    QCS6490 = 93,
    SM8845 = 97 // v81
};

enum class HtpArch: int{
    NONE=0,
    V68 =68,
    V69 =69,
    V73 =73,
    V75 =75,
    V79= 79,
    V81 =81,
};

enum class HtpGraphPrecision: int{
  kQuantized = 0,
  kFp16 = 1,
};