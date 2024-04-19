#include <pybind11/pybind11.h>
#include "cpu/CpuHook.h"
#include "cuda/GpuProfiler.h"
#include "hook/CFuncHook.h"
#include "hook/LocalHook/LocalHook.h"
#include "utils/Timer/Timer.h"
#include "utils/Lock/FileLock.h"
#include <iostream>

namespace py = pybind11;

void install_hook() {
    cfunc_hook::install_hook();
    local_hook::install_local_hooks();
}

void init_hook(pybind11::module& m) {
    m.def("install_hook", []() {
        cpu_hook::register_cpu_hook();
        gpu_profiler::register_gpu_hook();
        lock::do_func_in_one_process(install_hook);

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
}

PYBIND11_MODULE(Hook, m) {
    init_hook(m);
}
