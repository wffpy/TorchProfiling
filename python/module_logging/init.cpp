// #include "../../include/python_bind.h"
#include <pybind11/pybind11.h>
#include "CFuncHook.h"

namespace py = pybind11;

void init_hook(pybind11::module& m) {
    m.def("install_hook", []() {
        kernel_hook::install_hook();
    });
}

PYBIND11_MODULE(Hook, m) {
    init_hook(m);
}