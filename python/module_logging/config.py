import os
import configparser
from .logging import Logger

config = configparser.ConfigParser()
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(current_file_dir + "/config.ini"):
    Logger.info("config.ini found")
else:
    Logger.error("config.ini found")
config.read(current_file_dir + '/config.ini')

def get_config(section, key):
    '''get the configuration from config.ini'''
    return config[section][key]
