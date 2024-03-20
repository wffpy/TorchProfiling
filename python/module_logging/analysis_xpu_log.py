# from inspect import ArgSpec
# import sys
import os.path as osp

# import pandas as pd
import re
import prettytable as pt

# import pathlib


# import pandas.DataFrame as df
def parse_one_log(log_file_path):
    if not osp.exists(log_file_path):
        print("Log file {} does not exist".format(log_file_path))
        raise FileNotFoundError("log file {} doesn't find".format(log_file_path))
    lines = []
    with open(log_file_path, "r") as f:
        lines = f.readlines()

    table_data = []
    count = 0

    collecting = False
    iteration_begin = False
    xdnn_pytorch_op_list = []
    xdnn_pytorch_op_times = []
    total_kernel_consumed = 0
    xdnn_pytorch_op_dict = {}
    op_name = ""
    xdnn_pytorch_op_name = ""
    # collect the total time in the net
    total_time_dict = {}
    total_iteration_time = 0
    module_list = []
    module_total_kernel_consumed = 0
    for line in lines:
        if not iteration_begin:
            if "iteration" in line and "learning" in line and "loss" in line:
                iteration_begin = True
                print(">>>>>>>>>>>>start a collection iteration")
            continue

        # finish the collection of one iteration
        if iteration_begin:
            if "iteration" in line and "learning" in line and "loss" in line:
                print(">>>>>>>>>>>>end a collection iteration")
                break
        # print(line)

        if line.startswith("[BEGIN FORWARD]:") or line.startswith("[BEGINE BACKWARD]"):
            if len(module_list) > 0 and module_total_kernel_consumed > 0:
                data = {
                    "Module": "",
                    "Name": "",
                    "TOTAL TIME": "",
                    "MODULE TOTAL TIME": module_total_kernel_consumed,
                }
                table_data.append(data)

            module_list.append(line.rstrip("\n").split(":")[-1])
            module_total_kernel_consumed = 0

        if line.startswith("[END FORWARD]:") or line.startswith("[END BACKWARD]"):
            data = {
                "Module": "",
                "Name": "",
                "TOTAL TIME": "",
                "MODULE TOTAL TIME": module_total_kernel_consumed,
            }
            table_data.append(data)
            module_total_kernel_consumed = 0

            module_name = line.rstrip("\n").split(":")[-1]
            if module_name == module_list[-1]:
                module_list.pop()
            else:
                raise Exception("not find the module name in module list")

        if not collecting:
            if line.startswith("[START_SYMBOL]:"):
                collecting = True
                op_name = line.rstrip("\n").split(":")[-1]
                xdnn_pytorch_op_name = ""
            continue

        if line.startswith("[END_SYMBOL]:"):
            end_name = line.rstrip("\n").split(":")[-1]
            if end_name != op_name:
                raise Exception(
                    "end symbol name {} is not equal to start symbol name {}".format(
                        end_name, op_name
                    )
                )
            for op in xdnn_pytorch_op_list:
                if op not in xdnn_pytorch_op_dict:
                    raise Exception("op {} not in xdnn_pytorch_op_dict".format(op))
                xdnn_pytorch_op_times.append(xdnn_pytorch_op_dict[op])
            if len(xdnn_pytorch_op_times) == 0:
                xdnn_pytorch_op_times.append(0)

            module_name = ""
            if len(module_list) > 0:
                module_name = module_list[-1]

            data = {
                "Module": module_name,
                "Name": op_name,
                "XDNN_PYTORCH": xdnn_pytorch_op_list,
                "XDNN_PYTORCH_TIME": xdnn_pytorch_op_times,
                "TOTAL TIME": total_kernel_consumed,
            }
            table_data.append(data)

            if op_name in total_time_dict:
                total_time_dict[op_name] += total_kernel_consumed
            else:
                total_time_dict[op_name] = total_kernel_consumed

            op_name = ""
            xdnn_pytorch_op_list = []
            xdnn_pytorch_op_dict = {}
            total_kernel_consumed = 0
            xdnn_pytorch_op_times = []
            collecting = False
            # count += 1
            # if count == 100:
            #     break
            continue

        # collect the op profiling information for formal op:
        #  which will call the xdnn_pytorch op
        if line.startswith("xdnn_pytorch_") and "api::kXPU" in line:
            match = re.match(r"^([^()]*)\(", line)
            if not match:
                raise Exception("line {} doesn't match".format(line))
            xdnn_pytorch_op_name = match.group(1)
            xdnn_pytorch_op_list.append(xdnn_pytorch_op_name)
            xdnn_pytorch_op_dict[xdnn_pytorch_op_name] = 0
            continue

        # some kernel will call the xdnn op directly
        # for example: custom_ops.mha_varlen_fwd.default
        if line.startswith("gtest_") and "api::kXPU" in line:
            match = re.match(r"^([^()]*)\<", line)
            if not match:
                raise Exception("line {} doesn't match".format(line))
            xdnn_pytorch_op_name = match.group(1)
            xdnn_pytorch_op_list.append(xdnn_pytorch_op_name)
            xdnn_pytorch_op_dict[xdnn_pytorch_op_name] = 0
            continue

        if line.startswith("[XPURT_PROF]"):
            strs = line.split(" ")
            if xdnn_pytorch_op_name == "":
                xdnn_pytorch_op_name = op_name
                xdnn_pytorch_op_list.append(op_name)
                xdnn_pytorch_op_dict[xdnn_pytorch_op_name] = 0

            time = float(strs[-4]) / 1.45 / 1000000
            xdnn_pytorch_op_dict[xdnn_pytorch_op_name] = (
                time + xdnn_pytorch_op_dict[xdnn_pytorch_op_name]
            )
            total_kernel_consumed += time
            total_iteration_time += time
            module_total_kernel_consumed += total_kernel_consumed

    # sort table_date
    def take_total_time(data):
        """get the total time of each op"""
        return data["TOTAL TIME"]

    table = pt.PrettyTable(
        ["Module", "Name", "XDNN_PYTORCH", "XDNN_PYTORCH_TIME", "TOTAL TIME"]
    )
    for data in table_data:
        table.add_row(
            [
                data["Module"],
                data["Name"],
                data["XDNN_PYTORCH"],
                data["XDNN_PYTORCH_TIME"],
                data["TOTAL TIME"],
            ]
        )

    # summarize_table_data = []
    total_time_dict = sorted(total_time_dict.items(), key=lambda x: x[1], reverse=True)

    if total_iteration_time == 0:
        return

    table2 = pt.PrettyTable(["Name", "TOTAL TIME", "PERCENT"])
    for elem in total_time_dict:
        table2.add_row([elem[0], elem[1], elem[1] / total_iteration_time])

    table3 = pt.PrettyTable(["TOTAL TIME"])
    table3.add_row([total_iteration_time])

    return table, table2, table3


def parse_log(log_path, print_table=True):
    """
    Args:
        log_path: the path of log file
        print_table: whether to print table or not
    Function:
    analysis the log file generated with the following two environments:
        1. export XPURT_DISPATCH_MODE=PROFILING
        2. export XPUAPI_DEBUG=0x1001
    return:
    """

    table, table1, table2 = parse_one_log(log_path)
    if print_table:
        print(table)
        print(table1)
        print(table2)
