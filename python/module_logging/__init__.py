from .logger import *
from . import trace
from . import config
from .logger import PerformanceLogger, combined_context
from .percision import PercisionDebugger, percision_debugger
from .tensor_tracer import TensorTracer

tensor_tracer = TensorTracer()

cpp_extend = config.get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook
