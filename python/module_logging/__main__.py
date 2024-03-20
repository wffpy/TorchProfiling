from .tools import *
import os
import ctypes

script_dir = os.path.dirname(os.path.abspath(__file__))
cuda_mock_impl = ctypes.CDLL(f'{script_dir}/libcuda_mock_impl.so')


def main():
    print("This is mymodule's main function!")
    parse_log()


if __name__ == "__main__":
    main()
