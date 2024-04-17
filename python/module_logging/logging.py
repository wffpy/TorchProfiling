import os

log_level = os.environ.get("LOG_LEVEL")

def get_log_level():
    if log_level is None:
        return 0
    return int(log_level)


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class Logger(object):
    """
    implementation of Logger
    """

    @staticmethod
    def print_c(msg, color: Color):
        """print message with color"""
        print(color + msg + Color.END)

    @staticmethod
    def debug(msg):
        if get_log_level() >= 2:
            Logger.print_c("{}".format(msg), Color.RED)

    @staticmethod
    def info(msg):
        Logger.print_c("INFO: {}".format(msg), Color.GREEN)

    @staticmethod
    def warn(msg):
        if get_log_level() >= 1:
            Logger.print_c("WARN: {}".format(msg), Color.YELLOW)

    @staticmethod
    def error(msg):
        Logger.print_c("ERROR: {}".format(msg), Color.RED)
        exit(-1)
