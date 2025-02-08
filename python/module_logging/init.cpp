#include <pybind11/pybind11.h>
#include "cpu/CpuHook.h"
#include "cuda/GpuProfiler.h"
#include "hook/CFuncHook.h"
#include "dump/dump.h"
#include "hook/LocalHook/LocalHook.h"
#include "utils/ProfilingAccumulator/ProfilingAccumulator.h"
#include "utils/Timer/Timer.h"
#include "utils/Lock/FileLock.h"
#include "utils/Recorder/Recorder.h"
#include <iostream>

namespace py = pybind11;

void init_hook(pybind11::module& m) {
    m.def("install_hook", [](cfunc_hook::HookType hook_category) {
        // the following three funcs do nothing
        cpu_hook::register_cpu_hook();
        gpu_profiler::register_gpu_hook();

        cfunc_hook::install_hook(hook_category);
        local_hook::install_local_hooks();
    });

    m.def("enable_profiling", [](){
        timer::enable_timer();
    });

    m.def("init_timer", [](int64_t size) {
        timer::init_timer(size);
    });

    m.def("get_current_time", []() {
        return timer::get_time();
    });

    m.def("record_time", [](char* ph, char* name, char* tid) {
        timer::record_time(ph, name, tid);
    });

    m.def("set_timer_record_path", [](char* path) {
        std::string path_str(path);
        timer::set_record_path(path_str);
    });

    m.def("set_log_record_path", [](char* path) {
        std::string path_str(path);
        recorder::set_record_file(path_str);
    });

    m.def("record_log", [](char* str) {
        recorder::record(str);
    });

    m.def("write_to_file", []() {
        recorder::write_to_file();
    });

    m.def("enable_recorder", []() {
        recorder::enable_recorder();
    });

    m.def("close_recorder", []() {
        recorder::close_recorder();
    });

    m.def("cuda_profiler_start", []() {
        gpu_profiler::cupti_activity_init();
    });

    m.def("cuda_profiler_flush", []() {
        gpu_profiler::cupti_activity_flush();
    });

    m.def("cuda_profiler_end", []() {
        gpu_profiler::cupti_activity_finalize();
    });

    m.def("record_tensor", [](uint64_t ptr, int64_t size) {
        dump::record_tensor(ptr, size);
    });

    m.def("enable_profiling_accumulation", [](){
        profiling_accumulator::enable_profiling_accumulation();
    });

    m.def("disable_profiling_accumulation", [](){
        profiling_accumulator::disable_profiling_accumulation();
    });

    m.def("profiling_accumulation_start_iteration", [](int64_t iteration){
        profiling_accumulator::start_iteration(iteration);
    });

    m.def("profiling_accumulation_set_dump_json_path", [](char* path) {
        std::string path_str(path);
        profiling_accumulator::set_profiling_dump_file(path_str);
    });
}

PYBIND11_MODULE(Hook, m) {
    py::enum_<cfunc_hook::HookType>(m, "HookType")
        .value("kNONE", cfunc_hook::HookType::kNONE)
        .value("kDUMP", cfunc_hook::HookType::kDUMP)
        .value("kPROFILE", cfunc_hook::HookType::kPROFILE)
        .value("kACCUMULATE_KERNEL_TIME", cfunc_hook::HookType::kACCUMULATE_KERNEL_TIME)
        .export_values();
    init_hook(m);
}
