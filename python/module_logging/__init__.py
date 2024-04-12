from .logger import *
import configparser

# read config.ini
config = configparser.ConfigParser()
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(current_file_dir + "/config.ini"):
    print("config.ini found")
else:
    print("config.ini not found")
config.read(current_file_dir + '/config.ini')

# 获取cpp_extend配置
cpp_extend = config['database']['cpp_extend']

# just import Hook when cpp_extend is True
if cpp_extend == 'True':
    from . import Hook