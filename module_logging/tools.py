import argparse
from ast import arg
import pathlib
import sys
from .analysis_xpu_log import parse_log as parse_xpu_log
from .analysis_gpu_log import parse_log as parse_gpu_log
from .analysis_xpu_without_xdnn_pytorch import parse_log as parse_xpu_log2
# import analysis_gpu_log
# import analysis_xpu_log


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Module Logging command line tools.',
        formatter_class=argparse.RawTextHelpFormatter,
        )

    arg_parser.add_argument('--xpu_log', type=pathlib.Path, help="path to XPU log file")
    arg_parser.add_argument('--xpu_log_non_xdnn_pytorch', type=pathlib.Path, help="path to XPU log file")
    arg_parser.add_argument('--gpu_log', type=pathlib.Path, help="path to GPU log file")
    return arg_parser.parse_args()

def parse_log():
    args = parse_args()
    if args.xpu_log:
        parse_xpu_log(args.xpu_log)

    if args.gpu_log:
        parse_gpu_log(args.gpu_log)
    
    if args.xpu_log_non_xdnn_pytorch:
        parse_xpu_log2(args.xpu_log_non_xdnn_pytorch)

        
    