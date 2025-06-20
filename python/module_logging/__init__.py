from .configuration import get_config
from .debug import tensor_info_recorder
from .percision import PercisionDebugger, percision_debugger, tensor_tracer
from .perf import trace
from .perf.logger import PerformanceLogger
from .graph.graph_debugger import GraphDebugger
enable_cpp_extend = cpp_extend()

cpp_extend = get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook
