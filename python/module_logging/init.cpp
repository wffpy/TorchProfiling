#include <pybind11/pybind11.h>
#include "cpu/CpuHook.h"
#include "cuda/GpuProfiler.h"
#include "hook/CFuncHook.h"
#include "hook/LocalHook/LocalHook.h"
#include <iostream>

namespace py = pybind11;

void init_hook(pybind11::module& m) {
    m.def("install_hook", []() {
        cpu_hook::register_cpu_hook();
        gpu_profiler::register_gpu_hook();
        cfunc_hook::install_hook();
        local_hook::install_local_hooks();
    });
}

PYBIND11_MODULE(Hook, m) {
    init_hook(m);
}
