import configparser
import os

from ..utils.logging import Logger

config = configparser.ConfigParser()
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(current_file_dir + "/config.ini"):
    Logger.info("config.ini found, {}".format(current_file_dir + "/config.ini"))
else:
    Logger.error("config.ini found")
config.read(current_file_dir + "/config.ini")


def get_config(section, key):
    """get the configuration from config.ini"""
    return config[section][key]


def cpp_extend():
    """
    get the cpp_extend configuration from config.ini
    """
    cfg = get_config("database", "cpp_extend")
    if cfg == "True":
        return True
    return False
