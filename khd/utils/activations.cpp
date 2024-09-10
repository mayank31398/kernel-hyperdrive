#include "./dtypes.h"

fp32 sigmoid(const fp32 x) {
    fp32 output;
    if (x > 0) {
        output = 1 / (1 + expf(-x));
    } else if (x < 0) {
        output = expf(x);
        output = output / (1 + output);
    } else {
        output = 0;
    }

    return output;
}
