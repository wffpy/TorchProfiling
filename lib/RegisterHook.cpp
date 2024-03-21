#include "CFuncHook.h"

using namespace kernel_hook;

// REGISTERHOOK(xpu_wait, (void *)HookWrapper::local_xpu_wait,
//              (void **)&HookWrapper::instance()->origin_xpu_wait_);