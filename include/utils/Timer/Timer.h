#ifndef UTILS_TIMER_TIMER_H
#define UTILS_TIMER_TIMER_H
#include <iostream>
#include <chrono>

using namespace std::chrono;
namespace timer {
int64_t get_time();

void enable_timer();

void init_timer(int64_t size);

void set_record_path(const std::string& path);

high_resolution_clock::time_point record_time(std::string ph = "B", std::string name = "launch_async", std::string tid = "runtime", std::string cname = "yellow");

void record_flow_event(high_resolution_clock::time_point time, std::string ph = "s", std::string name = "event", std::string tid = "runtime", std::string cname = "yellow");

void record_time_pair(int64_t ns, std::string name, std::string tid, std::string cname="yellow");

int64_t get_time();

void record_duration();

int64_t get_duration();

}   // namespace timer
#endif