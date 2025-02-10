#include "utils/ProfilingAccumulator/ProfilingAccumulator.h"

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "utils/Log/Log.h"
#include "utils/Utils.h"

using namespace std;

namespace profiling_accumulator {

class ProfilingAccumulator {
   public:
    ProfilingAccumulator();
    ~ProfilingAccumulator();

    void start_iteration(int iteration);
    void accumulate_profiling_info(const string& kernel_name, int duration_ns, int device_cycles);
    void set_dump_json_path(const string& json_path);

    static void enable_accumulator();
    static void disable_accumulator();

    static bool enable;

   private:
    void print_last_iteration();
    void print_new_kernel_name_to_idx();
    void dump_to_json();

    string dump_json_path;
    int current_iteration;
    int printed_idx;
    unordered_map<string, int> kernel_name_to_idx;
    map<int, map<int, tuple<int64_t, int64_t>>> profiling_info;
};

bool ProfilingAccumulator::enable = false;

ProfilingAccumulator::ProfilingAccumulator() : current_iteration(1), printed_idx(-1) {
    profiling_info[current_iteration] = {};
}

ProfilingAccumulator::~ProfilingAccumulator() {
    if (enable) {
        print_last_iteration();
        print_new_kernel_name_to_idx();
        dump_to_json();
    }
}

void ProfilingAccumulator::start_iteration(int iteration) {
    if (enable) {
        CHECK(iteration >= current_iteration, "Iteration must be strictly increasing");
        print_last_iteration();
        current_iteration = iteration + 1;
        profiling_info[iteration] = {};
    }
}

void ProfilingAccumulator::accumulate_profiling_info(const string& kernel_name, int duration_ns, int device_cycles) {
    if (enable) {
        auto& info = profiling_info[current_iteration];
        int kernel_idx = -1;
        if (kernel_name_to_idx.find(kernel_name) == kernel_name_to_idx.end()) {
            kernel_idx = kernel_name_to_idx.size();
            kernel_name_to_idx[kernel_name] = kernel_idx;
            info[kernel_idx] = make_tuple<int64_t, int64_t>(0, 0);
        } else {
            kernel_idx = kernel_name_to_idx[kernel_name];
        }
        get<0>(info[kernel_idx]) += duration_ns;
        get<1>(info[kernel_idx]) += device_cycles;
    }
}

void ProfilingAccumulator::set_dump_json_path(const string& json_path) { dump_json_path = json_path; }

void ProfilingAccumulator::enable_accumulator() { enable = true; }

void ProfilingAccumulator::disable_accumulator() { enable = false; }

void ProfilingAccumulator::print_last_iteration() {
    if (enable) {
        auto& info = profiling_info[current_iteration];
        cout << "[Accum] iter " << current_iteration << ": ";
        for (const auto& [kernel_idx, stats] : info) {
            int64_t duration_ns = get<0>(stats);
            int64_t device_cycles = get<1>(stats);
            cout << "(" << kernel_idx << ", " << duration_ns << ", " << device_cycles << "), ";
        }
        cout << endl;

        if (current_iteration == 1) {
            print_new_kernel_name_to_idx();
        }
    }
}

void ProfilingAccumulator::print_new_kernel_name_to_idx() {
    if (enable && printed_idx + 1 < kernel_name_to_idx.size()) {
        unordered_map<int, string> idx_to_kernel_name;
        for (const auto& [kernel_name, idx] : kernel_name_to_idx) {
            if (idx > printed_idx) {
                idx_to_kernel_name[idx] = kernel_name;
            }
        }
        cout << "[Accum] kernel_name_to_idx: ";
        for (int idx = printed_idx + 1; idx < kernel_name_to_idx.size(); ++idx) {
            cout << idx_to_kernel_name[idx] << ", " << idx << "; ";
        }
        cout << endl;
        printed_idx = kernel_name_to_idx.size() - 1;
    }
}

void ProfilingAccumulator::dump_to_json() {
    if (enable && !dump_json_path.empty()) {
        try {
            unordered_map<int, string> idx_to_kernel_name;
            for (const auto& [kernel_name, idx] : kernel_name_to_idx) {
                idx_to_kernel_name[idx] = kernel_name;
            }
            ofstream outfile(dump_json_path);
            outfile << "[\n";
            bool first = true;
            for (const auto& [iteration, info] : profiling_info) {
                if (!first) {
                    outfile << ",\n";
                } else {
                    first = false;
                }
                outfile << "{\n"
                        << "\"iteration\": " << iteration << ",\n"
                        << "\"kernels\": [\n";
                bool first_kernel = true;
                for (const auto& [kernel_idx, stats] : info) {
                    if (!first_kernel) {
                        outfile << ",\n";
                    } else {
                        first_kernel = false;
                    }
                    int64_t duration_ns = get<0>(stats);
                    int64_t device_cycles = get<1>(stats);
                    outfile << "{\n"
                            << "\"kernel_name\": \"" << idx_to_kernel_name[kernel_idx] << "\",\n"
                            << "\"duration_ns\": " << duration_ns << ",\n"
                            << "\"device_cycles\": " << device_cycles << "\n"
                            << "}";
                }
                outfile << "\n]\n}\n";
            }
            outfile << "]";
        } catch (const std::exception& e) {
            ELOG() << "Failed to dump profiling info: " << e.what() << "\n";
        } catch (...) {
            ELOG() << "Failed to dump profiling info: unknown error"
                   << "\n";
        }
    }
}

typedef utils::Singleton<ProfilingAccumulator> ProfilingAccumulatorSingleton;

void enable_profiling_accumulation() { ProfilingAccumulatorSingleton::instance().get_elem()->enable_accumulator(); }

void disable_profiling_accumulation() { ProfilingAccumulatorSingleton::instance().get_elem()->disable_accumulator(); }

void start_iteration(int iteration) {
    ProfilingAccumulatorSingleton::instance().get_elem()->start_iteration(iteration);
}

void accumulate_profiling_info(const string& kernel_name, int duration_ns, int device_cycles) {
    ProfilingAccumulatorSingleton::instance().get_elem()->accumulate_profiling_info(kernel_name, duration_ns,
                                                                                    device_cycles);
}

void set_profiling_dump_file(const string& file_path) {
    ProfilingAccumulatorSingleton::instance().get_elem()->set_dump_json_path(file_path);
}

}  // namespace profiling_accumulator
