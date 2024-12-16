#ifndef BACKTRACE_H
#define BACKTRACE_H
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

namespace trace {
// reference: https://man7.org/linux/man-pages/man3/backtrace.3.html
class Tracer {
public:
    Tracer(std::string name);
    ~Tracer();
private:
    void trace();
    void print();
    int64_t max_depth;
    int64_t real_size;
    // start time of the function
    std::chrono::high_resolution_clock::time_point start;
    bool enable_duration;
    const std::string func_name;
    std::vector<void*> stack;
};

}   // namespace trace
#endif

