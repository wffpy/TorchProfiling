#include <pybind11/pybind11.h>
#include "cpu/CpuHook.h"
#include "cuda/GpuProfiler.h"
#include "hook/CFuncHook.h"
#include "hook/LocalHook/LocalHook.h"
#include "utils/Timer/Timer.h"
#include "utils/Lock/FileLock.h"
#include "utils/Recorder/Recorder.h"
#include <iostream>

namespace py = pybind11;

void init_hook(pybind11::module& m) {
    m.def("install_hook", []() {
        cpu_hook::register_cpu_hook();
        gpu_profiler::register_gpu_hook();
        cfunc_hook::install_hook();
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

}

PYBIND11_MODULE(Hook, m) {
    init_hook(m);
}
