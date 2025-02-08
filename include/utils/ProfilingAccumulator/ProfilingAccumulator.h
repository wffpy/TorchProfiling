#ifndef UTILS_PROFILING_ACCUMULATOR_H
#define UTILS_PROFILING_ACCUMULATOR_H

#include <string>

namespace profiling_accumulator {

void enable_profiling_accumulation();

void disable_profiling_accumulation();

void start_iteration(int iteration_number);

void accumulate_profiling_info(const std::string& kernel_name, int duration_ns, int device_cycles);

void set_profiling_dump_file(const std::string& file_path);

}  // namespace profiling_accumulator

#endif  // UTILS_PROFILING_ACCUMULATOR_H
