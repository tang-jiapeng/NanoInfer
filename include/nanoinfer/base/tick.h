/**
 * @file tick.h
 * @brief 简易计时宏 TICK / TOCK
 */
#ifndef NANO_INFER_TICK_H
#define NANO_INFER_TICK_H
#include <chrono>
#include <iostream>

#ifndef __ycm__
/// @brief 开始计时
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
/// @brief 结束计时并打印秒数
#define TOCK(x)                                                       \
    printf("%s: %lfs\n", #x,                                          \
           std::chrono::duration_cast<std::chrono::duration<double>>( \
               std::chrono::steady_clock::now() - bench_##x)          \
               .count());
#else
#define TICK(x)
#define TOCK(x)
#endif
#endif  // NANO_INFER_TICK_H