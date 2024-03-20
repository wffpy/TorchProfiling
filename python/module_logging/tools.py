import argparse
from ast import arg
import pathlib
import sys
from .analysis_xpu_log import parse_log as parse_xpu_log
from .analysis_gpu_log import parse_log as parse_gpu_log
from .analysis_xpu_without_xdnn_pytorch import parse_log as parse_xpu_log2
import prettytable as pt

# import analysis_gpu_log
# import analysis_xpu_log


def parse_args():
    """
    Parse the input arguments

    Returns:
        void
    """

    arg_parser = argparse.ArgumentParser(
        description="Module Logging command line tools.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # arg_parser.add_argument("--c", type=pathlib.Path, help="path to XPU log file")
    arg_parser.add_argument(
        "--compare", action="store_true", help="compares the two log files"
    )
    arg_parser.add_argument("--xpu_log", type=pathlib.Path, help="path to XPU log file")
    arg_parser.add_argument(
        "--xpu_log_non_xdnn_pytorch", type=pathlib.Path, help="path to XPU log file"
    )
    arg_parser.add_argument("--gpu_log", type=pathlib.Path, help="path to GPU log file")
    return arg_parser.parse_args()


def parse_log():
    """
    Parse the input arguments and run the analysis.
    Returns:
        void
    """

    args = parse_args()
    gpu_cost_list = []
    xpu_cost_list = []

    if args.xpu_log:
        parse_xpu_log(args.xpu_log, not args.compare)

    if args.gpu_log:
        gpu_cost_list = parse_gpu_log(args.gpu_log, not args.compare)

    if args.xpu_log_non_xdnn_pytorch:
        xpu_cost_list = parse_xpu_log2(args.xpu_log_non_xdnn_pytorch, not args.compare)

    if args.compare:
        gpu_cost_list_len = len(gpu_cost_list)
        xpu_cost_list_len = len(xpu_cost_list)
        if gpu_cost_list_len > 0 and xpu_cost_list_len > 0:
            gpu_index = 0
            xpu_index = 0
            gpu_step_fwd = False
            xpu_step_fwd = False
            table = pt.PrettyTable(["Module", "gpu", "xpu", "diff"])
            summary = {}

            while gpu_index < gpu_cost_list_len:
                gpu_cost_tuple = gpu_cost_list[gpu_index]
                if xpu_index < xpu_cost_list_len:
                    xpu_cost_tuple = xpu_cost_list[xpu_index]

                    if gpu_cost_tuple[0] == xpu_cost_tuple[0]:
                        module_name = gpu_cost_tuple[0]
                        gpu_data = gpu_cost_tuple[1]
                        xpu_data = xpu_cost_tuple[1]
                        table.add_row(
                            [module_name, gpu_data, xpu_data, xpu_data - gpu_data]
                        )
                        # if summary.has_key(module_name):
                        if module_name in summary.keys():
                            summary[module_name] += xpu_data - gpu_data
                        else:
                            summary[module_name] = xpu_data - gpu_data
                        if gpu_index < gpu_cost_list_len:
                            gpu_index += 1
                        if xpu_index < xpu_cost_list_len:
                            xpu_index += 1
                        gpu_step_fwd = False
                        xpu_step_fwd = False
                    else:
                        if gpu_index +1 < gpu_cost_list_len and gpu_cost_list[gpu_index + 1][0] == xpu_cost_tuple[0]:
                            gpu_index += 1
                        elif xpu_index + 1 < xpu_cost_list_len and gpu_cost_tuple[0] == xpu_cost_list[xpu_index + 1][0]:
                            xpu_index += 1
                        else:
                            gpu_module_name = gpu_cost_tuple[0]
                            gpu_data = gpu_cost_tuple[1]
                            table.add_row([gpu_module_name, gpu_data, 0, 0])
                            if gpu_index < gpu_cost_list_len:
                                gpu_index += 1
                else:
                    gpu_module_name = gpu_cost_tuple[0]
                    gpu_data = gpu_cost_tuple[1]
                    table.add_row([gpu_module_name, gpu_data, 0, 0])
                    if gpu_index < gpu_cost_list_len:
                        gpu_index += 1
            print(table)

            summary = sorted(summary.items(), key=lambda x: x[1], reverse=True)
            sum_table = pt.PrettyTable(["Module", "total cost"])
            # for key in summary.keys():
            for elem in summary:
                sum_table.add_row([elem[0], elem[1]])

                # print(key, summary[key])
            print(sum_table)
