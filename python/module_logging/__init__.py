from .configuration import get_config
from .debug import tensor_info_recorder
from .percision import PercisionDebugger, percision_debugger, tensor_tracer
from .perf import trace
from .perf.logger import PerformanceLogger
from .perf.cpu_perf import CpuPerformanceLogger

cpp_extend = get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook
