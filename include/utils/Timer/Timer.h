#ifndef UTILS_TIMER_TIMER_H
#define UTILS_TIMER_TIMER_H
#include <iostream>

namespace timer {
int64_t get_time();

void enable_timer();

void init_timer(int64_t size);

void record_time(std::string ph = "B", std::string name = "launch_async", std::string tid = "runtime", std::string cname = "yellow");

int64_t get_time();

void record_duration();

int64_t get_duration();

}   // namespace timer
#endif