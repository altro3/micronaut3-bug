#ifndef SPECULATIVE_TYPES_H
#define SPECULATIVE_TYPES_H

#include <stdint.h>

struct SpecSharedState {
    bool rejected;
    float max_logit;
    float total_sum;
    float norm_sum;
    int32_t accepted_count;
    float warp_exchange[32];
};

#endif
