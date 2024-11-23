from .perf import config
from .perf.logger import PerformanceLogger
from .percision import PercisionDebugger, percision_debugger
from .percision import tensor_tracer
from .perf import trace
from .debug import tensor_info_recorder

cpp_extend = config.get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook