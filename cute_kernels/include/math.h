#pragma once

#include "dtypes/all.h"

inline bool check_power_of_2(const uint32 &n) { return ((n - 1) & n == 0) && n != 0; }
