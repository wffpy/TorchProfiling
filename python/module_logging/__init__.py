from .logger import *
from . import trace
from . import config

cpp_extend = config.get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook
