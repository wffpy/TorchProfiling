#include <pybind11/pybind11.h>
#include "CFuncHook.h"
#include <iostream>

namespace py = pybind11;

void init_hook(pybind11::module& m) {
    m.def("install_hook", []() {
        std::cout << "install hook!!!!!!!!!!!!!!!!" << std::endl;
        kernel_hook::install_hook();
    });
}

PYBIND11_MODULE(Hook, m) {
    init_hook(m);
}
