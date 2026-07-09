#pragma once

#include <stdio.h>

#ifndef LOG_INF
#define LOG_INF(...) fprintf(stdout, __VA_ARGS__)
#endif

#ifndef LOG_ERR
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif
