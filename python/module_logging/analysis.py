from asyncio import get_event_loop
from locale import locale_alias
from lzma import FORMAT_ALONE
import os.path as osp
from ossaudiodev import SOUND_MIXER_ALTPCM
import re
from typing_extensions import Self
from xml.etree.ElementTree import C14NWriterTarget
import prettytable as pt
from .logging import Logger
from enum import Enum, auto
from textwrap import fill


class STATE(Enum):
    BEGIN = auto()
    MODULE = auto()
    FORMAL = auto()
    OP = auto()
    DISTOP = auto()
    STOP = auto()


class Block(object):
    def __init__(self, name=""):
        self.op_list = []
        self.name = name
        self.time = 0

    def add_op(self, op):
        self.op_list.append(op)
        self.time += op.get_time()

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_time(self):
        return self.time

    def clear(self):
        self.op_list = []
        self.name = ""
        self.time = 0

    def get_op_list(self):
        return self.op_list


class LocalModule(object):
    def __init__(self, name: str = "") -> None:
        self._name_ = name
        self.element_list = []

    def get_name(self):
        return self._name_

    def add_elem(self, elem):
        self.element_list.append(elem)

    def get_elem_list(self):
        return self.element_list

    def get_elem_num(self):
        return len(self.element_list)

    def get_elem(self, index):
        return self.element_list[index]

    def dfs_traverse(self):
        _list = []
        visited = set()
        final_list = []
        for elem in self.element_list:
            _list.append(elem)
            while len(_list) > 0:
                local_elem = _list[-1]
                # LocalModule
                if isinstance(local_elem, LocalModule):
                    has_unvisited_son = False
                    for elem in local_elem.get_elem_list():
                        if elem not in visited:
                            _list.append(elem)
                            has_unvisited_son = True
                            break
                    if not has_unvisited_son:
                        visited.add(local_elem)
                        _list.pop()
                #  OpInfoBase
                else:
                    visited.add(local_elem)
                    final_list.append(local_elem)
                    _list.pop()
        return final_list

    def has_sub_module(self):
        for elem in self.element_list:
            if isinstance(elem, LocalModule):
                return True
        return False

    def get_sub_modules(self):
        module_list = []
        for elem in self.element_list:
            if isinstance(elem, LocalModule):
                module_list.append(elem)
        return module_list


class OpInfoBase(object):
    def __init__(self, name="", m_name="", time=0) -> None:
        self._name_ = name
        self.module_name = m_name
        self._time_ = time

    def set_time(self, time):
        # ms
        self._time_ += time
    
    def get_name(self):
        return self._name_

    def get_time(self):
        # ms
        return self._time_
    
    def get_module_name(self):
        return self.module_name
    
# record AtenOp info
class AtenOp(OpInfoBase):
    def __init__(self, name="", m_name="", time=0) -> None:
        super().__init__(name, m_name, time)

# record Distribution Op info
class DistOp(OpInfoBase):
    def __init__(self, name="", m_name="", time=0, d_bytes=0) -> None:
        super().__init__(name, m_name, time)
        self._bytes_ = d_bytes
    def set_bytes(self, bts):
        self._bytes_ = bts
    
    def get_bytes(self):
        return self._bytes_
    
    def get_bw(self):
        # GB/s
        if self._time_ == 0:
            return 0
        return self._bytes_ / self._time_ / 1000000

class OpSummary(object):
    def __init__(self, time) -> None:
        self.call_count = 1
        # ms
        self.total_time = time
        self.min = time
        self.max = time

    def add_time(self, time):
        self.total_time += time
        self.call_count += 1
        if time > self.max:
            self.max = time
        elif time < self.min:
            self.min = time

    def get_total_time(self):
        return self.total_time

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

    def get_avg(self):
        return self.total_time / self.call_count

    def get_call_count(self):
        return self.call_count

class AtenOpSummary(OpSummary):
    def __init__(self, time) -> None:
        super().__init__(time)

class DistOpSummary(OpSummary):
    def __init__(self, time, bytes) -> None:
        super().__init__(time)
        self.total_bytes = bytes
    
    def add_bytes(self, byt):
        self.total_bytes += byt
    
    def get_total_bytes(self):
        return self.total_bytes

    def get_avg_bw(self):
        return self.total_bytes / self.total_time / 1000000

class ModuleStack(object):
    def __init__(self) -> None:
        self.module_list = []

    def push(self, module):
        self.module_list.append(module)

    def pop(self):
        self.module_list.pop()

    def depth(self):
        return len(self.module_list)

    def top(self):
        return self.module_list[-1]


class Analyzer:
    def __init__(self, path):
        """
        args:
            path: the path of log file
        """
        if not osp.exists(path):
            raise FileNotFoundError("log file {} doesn't find".format(path))
        self.log_path = path
        # current module name
        self.current_m_name = ""
        self.collection_state = STATE.FORMAL
        self.current_op_name = ""
        self.current_op = None
        self.current_module = None
        self.stack = ModuleStack()
        self.op_or_module = []
        self.total = 0

    # def identify_step_beign_or_end(self, line: str):
    #     """
    #     args:
    #         line: a string of log file
    #     return:
    #         bool
    #     """
    #     if "iteration" in line and "learning" in line and "loss" in line:
    #         if self.collection_state == STATE.BEGIN:
    #             Logger.debug("Step Begin")
    #             self.collection_state = STATE.FORMAL
    #         else:
    #             Logger.debug("Step End")
    #             self.collection_state = STATE.STOP
    #         return True
    #     return False

    def identify_module_begin(self, line: str):
        """
        args:
            line: a string of log file
        return:
            bool
        Function:
        1. identify the module by following symbols: [BEGIN FORWARD], [END FORWARD], [BEGIN BACKWARD], [END BACKWARD]o
        2. if there is no module in this iteration, return None
        """
        if (
            (
                self.collection_state == STATE.FORMAL
                or self.collection_state == STATE.MODULE
            )
            and ("[BEGIN FORWARD]" in line or "[BEGIN BACKWARD]" in line)
        ):
            # if self.current_module is not None:
            #     self.stack.push(self.current_module)
            self.current_m_name = line.rstrip("\n").split(":")[-1]
            self.current_module = LocalModule(self.current_m_name)
            self.collection_state = STATE.MODULE
            Logger.debug("Module Begin: {}".format(self.current_m_name))
            self.stack.push(self.current_module)
            return True
        return False

    def identify_module_end(self, line: str):
        """
        args:
            line: a string of log file
        return:
            module_name: the name of module
        Function:
        1. identify the module by following symbols: [BEGIN FORWARD], [END FORWARD], [BEGIN BACKWARD], [END BACKWARD]o
        2. if there is no module in this iteration, return None
        """
        if self.collection_state == STATE.MODULE and (
            "[END FORWARD]" in line or "[END BACKWARD]" in line
        ):
            Logger.debug("Module End: {}".format(self.current_m_name))
            temp_module = self.current_module
            self.stack.pop()
            self.current_module = self.stack.top() if 0 < self.stack.depth() else None
            self.current_m_name = (
                self.current_module.get_name() if self.current_module else ""
            )
            self.collection_state = (
                STATE.FORMAL if self.current_module is None else STATE.MODULE
            )
            if self.current_module is None:
                self.op_or_module.append(temp_module)
            else:
                self.current_module.add_elem(temp_module)
            # if 0 < self.stack.depth():
            #     self.stack.pop()

            return True
        return False

    def identify_op_start(self, line: str):
        pass

    def identify_op_end(self, line: str):
        pass

    def identify_op_time(self, line: str):
        if (
            self.collection_state == STATE.OP
            or self.collection_state == STATE.DISTOP
        ) and "[XPURT_PROF]" in line:
            Logger.debug("Op Time")
            if self.current_op:
                self.current_op.set_time(float(line.split(" ")[-2]) / 1000000)
            return True
        return False

    def get_total(self):
        return self.total

    def analysis(self):
        pass

    def get_op_list(self):
        final_list = []
        for elem in self.op_or_module:
            if isinstance(elem, LocalModule):
                for local_elem in elem.dfs_traverse():
                    final_list.append(local_elem)
            elif isinstance(elem, OpInfoBase):
                final_list.append(elem)
        return final_list

    def gen_detail_table(self):
        pass

    def gen_summary_table(self):
        pass

    def gen_total_time_table(self):
        """
        Function:
        1. generate a summary table with the following format:
        Total Time(ms)
        --------------------------------------------------
        10000.123456789
        """
        table = pt.PrettyTable(["Total Time(ms)"])
        table.add_row([self.get_total()])
        return table

    def get_modules(self):
        for elem in self.op_or_module:
            if isinstance(elem, LocalModule):
                yield elem

class AtenOpAnalyzer(Analyzer):
    def __init__(self, path):
        super().__init__(path)

    def identify_op_start(self, line: str):
        """
        args:
            line: a string of log file
        return:
            bool
        Function:
        1. identify the op by following symbols: [START_SYMBOL], [END_SYMBOL]
        2. if there is no op in this iteration, return None
        """
        if (
            self.collection_state == STATE.FORMAL
            or self.collection_state == STATE.MODULE
        ) and "[START_SYMBOL]" in line:
            Logger.debug("Op Start")
            self.collection_state = STATE.OP
            self.current_op_name = line.rstrip("\n").split(":")[-1].replace("_", " ")
            self.current_op = AtenOp(self.current_op_name, self.current_m_name)
            return True
        return False

    def identify_op_end(self, line: str):
        if self.collection_state == STATE.OP and "[END_SYMBOL]" in line:
            Logger.debug("Op End")
            self.total += self.current_op.get_time()
            if self.current_module is not None:
                self.current_module.add_elem(self.current_op)
            else:
                self.op_or_module.append(self.current_op)
            self.current_op = None
            self.collection_state = (
                STATE.FORMAL if self.current_module is None else STATE.MODULE
            )
            return True
        return False

    def analysis(self):
        lines = []
        with open(self.log_path, "r") as f:
            lines = f.readlines()
        line_index =0
        for line in lines:
            Logger.debug("Line {}: {}".format(line_index, line))
            line_index += 1
            if self.identify_module_begin(line):
                continue
            elif self.identify_module_end(line):
                continue
            elif self.identify_op_start(line):
                continue
            elif self.identify_op_end(line):
                continue
            else:
                self.identify_op_time(line)
 
    def gen_detail_table(self):
        """
        Function:
        1. generate a table with the following format:
        Module                     Aten Op               Time(ms)
        ----------------------------------------------------------
        Conv2d                     conv2d                10.3456789
        BatchNorm2d                 batch_norm            1.2345678
        """
        # _list = []
        # final_list = self.get_op_list()
        for elem in self.op_or_module:
            if isinstance(elem, LocalModule):
                for local_elem in elem.dfs_traverse():
                    final_list.append(local_elem)
            elif isinstance(elem, OpInfoBase):
                final_list.append(elem)

        table = pt.PrettyTable(["Module", "Aten Op", "Time"])
        for elem in final_list:
            table.add_row(
                [
                    fill(elem.get_module_name(), width=50),
                    elem.get_name(),
                    elem.get_time(),
                ]
            )
        table.set_style(pt.DEFAULT)
        table.align = "l"
        return table

    def gen_summary_table(self):
        """
        Function:
        1. generate a summary table with the following format:
        Op                          Max Time(ms)   Min Time(ms)    Avg Time(ms)    Total Time(ms)  Count
        --------------------------------------------------------------------------------------------
        conv2d                      10.3456789     1.2345678       1.2345678         123456789   1000
        """
        final_list = self.get_op_list()
        op_dict = {}
        for elem in final_list:
            k = elem.get_name()
            v = elem.get_time()
            if k in op_dict:
                op_dict[k].add_time(v)
            else:
                op_summary = AtenOpSummary(v)
                op_dict[k] = op_summary

        op_list = sorted(
            op_dict.items(), key=lambda x: x[1].get_total_time(), reverse=True
        )

        table = pt.PrettyTable(
            [
                "Op",
                "Max Time(ms)",
                "Min Time(ms)",
                "Avg Time(ms)",
                "Total Time(ms)",
                "Count",
                "Percent(%)",
            ]
        )
        for op in op_list:
            percent = op[1].get_total_time() / self.get_total()
            table.add_row(
                [
                    fill(op[0], width=40),
                    op[1].get_max(),
                    op[1].get_min(),
                    op[1].get_avg(),
                    op[1].get_total_time(),
                    op[1].get_call_count(),
                    percent,
                ]
            )
        table.align = "l"
        return table

class DistAnalyzer(Analyzer):
    def __init__(self, path):
        super().__init__(path)

    def identify_op_start(self, line: str):
        """
        args:
            line: a string of log file
        return:
            bool
        Function:
        1. identify the op by following symbols: [START_SYMBOL], [END_SYMBOL]
        2. if there is no op in this iteration, return None
        """
        if (
            self.collection_state == STATE.FORMAL
            or self.collection_state == STATE.MODULE
        ) and "[DIST START_SYMBOL]" in line:
            Logger.debug("DIST Op Start")
            self.collection_state = STATE.DISTOP
            self.current_op_name = line.rstrip("\n").split(":")[-1].replace("_", " ")
            self.current_op = DistOp(self.current_op_name, self.current_m_name)
            return True
        return False

    def identify_op_end(self, line: str):
        if self.collection_state == STATE.DISTOP and "[DIST END_SYMBOL]" in line:
            Logger.debug("DIST Op End")
            self.total += self.current_op.get_time()
            if self.current_module is not None:
                self.current_module.add_elem(self.current_op)
            else:
                self.op_or_module.append(self.current_op)
            self.current_op = None
            self.collection_state = (
                STATE.FORMAL if self.current_module is None else STATE.MODULE
            )
            return True
        return False

    def identify_dist_bytes(self, line:str):
        if self.collection_state == STATE.DISTOP and "[DIST BYTES]" in line:
            Logger.debug("DIST Bytes")
            if self.current_op:
                self.current_op.set_bytes(int(line.split(" ")[-2]))
            return True
        return False

    def analysis(self):
        '''
        ananlyis the distributed ops
        '''
        lines = []
        with open(self.log_path, "r") as f:
            lines = f.readlines()
        line_index =0
        for line in lines:
            Logger.debug("Line {}: {}".format(line_index, line))
            line_index += 1
            if self.identify_module_begin(line):
                continue
            elif self.identify_module_end(line):
                continue
            elif self.identify_op_start(line):
                continue
            elif self.identify_op_end(line):
                continue
            elif self.identify_dist_bytes(line):
                continue
            else:
                self.identify_op_time(line)

    def gen_detail_table(self):
        """
        Function:
        1. generate a table with the following format:
        Module                     Aten Op        Bytes   Time(ms)     BW(GB/s)           Percent
        ----------------------------------------------------------------------------
        Net                       all_reduce      1000     10     10.3456789         10.3456789/20
        """
        final_list = []
        for elem in self.op_or_module:
            if isinstance(elem, LocalModule):
                for local_elem in elem.dfs_traverse():
                    final_list.append(local_elem)
            elif isinstance(elem, OpInfoBase):
                final_list.append(elem)

        table = pt.PrettyTable(["Module", "Dist Op", "Bytes", "Time(ms)", "BW(GB/s)", "Percent(BW/20)"])
        for elem in final_list:
            table.add_row(
                [
                    fill(elem.get_module_name(), width=50),
                    elem.get_name(),
                    elem.get_bytes(),
                    elem.get_time(),
                    elem.get_bw(),
                    elem.get_bw() / 20
                ]
            )
        table.set_style(pt.DEFAULT)
        table.align = "l"
        return table
    
    def gen_summary_table(self):
        final_list = self.get_op_list()
        op_dict = {}
        for elem in final_list:
            k = elem.get_name()
            v = elem.get_time()
            by = elem.get_bytes()
            if v == 0:
                continue
            if k in op_dict:
                op_dict[k].add_time(v)
                op_dict[k].add_bytes(by)
            else:
                op_summary = DistOpSummary(v, by)
                op_dict[k] = op_summary
        op_list = sorted(
            op_dict.items(), key=lambda x: x[1].get_total_time(), reverse=True
        )

        table = pt.PrettyTable(
            [
                "Op",
                "Total Time(ms)",
                "Total Bytes",
                "Avg Badnwidth(GB/s)",
                "Percent(%)"
            ]
        )
        for op in op_list:
            percent = op[1].get_avg_bw() / 20 
            table.add_row(
                [
                    fill(op[0], width=40),
                    op[1].get_total_time(),
                    op[1].get_total_bytes(),
                    op[1].get_avg_bw(),
                    percent,
                ]
            )
        table.align = "l"
        return table
        
 
def count_module(op_or_moudle: list):
    counter = 0
    for elem in op_or_moudle:
        if isinstance(elem, LocalModule):
            counter += 1
    return counter


def can_compare_module(lhs: LocalModule, rhs: LocalModule):
    lhs_sub_modules = lhs.get_sub_modules()
    rhs_sub_modules = rhs.get_sub_modules()
    if len(lhs_sub_modules) != len(rhs_sub_modules):
        return False
    for lhs_elem, rhs_elem in zip(lhs_sub_modules, rhs_sub_modules):
        if lhs_elem.get_name() != rhs_elem.get_name():
            return False
    return True


def compare_module(lhs: LocalModule, rhs: LocalModule):
    """
    Function:
    1. compare the two modules and return a table with the following format:
    """
    if can_compare_module(lhs, rhs):
        son_m_num = len(lhs.get_sub_modules())
        final_lhs_block_list = []
        final_rhs_block_list = []
        lhs_block_list = []
        rhs_block_list = []
        for i in range(son_m_num + 1):
            block = Block(lhs.get_name())
            lhs_block_list.append(block)
            block = Block(rhs.get_name())
            rhs_block_list.append(block)
        block_index = 0
        for elem in lhs.get_elem_list():
            if isinstance(elem, OpInfoBase):
                lhs_block_list[block_index].add_op(elem)
            elif isinstance(elem, LocalModule):
                block_index += 1
        block_index = 0
        for elem in rhs.get_elem_list():
            if isinstance(elem, OpInfoBase):
                rhs_block_list[block_index].add_op(elem)
            elif isinstance(elem, LocalModule):
                block_index += 1

        lhs_sub_modules = lhs.get_sub_modules()
        rhs_sub_modules = rhs.get_sub_modules()
        lhs_son_m_list_list = []
        rhs_son_m_list_list = []
        for lhs_son_m, rhs_son_m in zip(lhs_sub_modules, rhs_sub_modules):
            lhs_son_m_block_list, rhs_son_m_block_list = compare_module(
                lhs_son_m, rhs_son_m
            )
            lhs_son_m_list_list.append(lhs_son_m_block_list)
            rhs_son_m_list_list.append(rhs_son_m_block_list)

        for index in range(son_m_num + 1):
            final_lhs_block_list.append(lhs_block_list[index])
            final_rhs_block_list.append(rhs_block_list[index])
            if index < son_m_num:
                final_lhs_block_list.extend(lhs_son_m_list_list[index])
                final_rhs_block_list.extend(rhs_son_m_list_list[index])
        return final_lhs_block_list, final_rhs_block_list
    else:
        lhs_op_list = lhs.dfs_traverse()
        rhs_op_list = rhs.dfs_traverse()
        lhs_block = Block(lhs.get_name())
        for elem in lhs_op_list:
            lhs_block.add_op(elem)

        rhs_block = Block(rhs.get_name())
        for elem in rhs_op_list:
            rhs_block.add_op(elem)
        return {lhs_block}, {rhs_block}


def merge_block(lhs: Block, rhs: Block):
    """
    Function:
    """
    if lhs.get_name() != rhs.get_name():
        Logger.error("The name of two blocks is not the same")
    lhs_op_list = lhs.get_op_list()
    rhs_op_list = rhs.get_op_list()
    max_len = (
        len(lhs_op_list) if len(lhs_op_list) > len(rhs_op_list) else len(rhs_op_list)
    )
    table = pt.PrettyTable(
        [
            fill("Module Name", width=200),
            fill("GPU Op", width=100),
            fill("GPU Time(ms)", width=30),
            fill("XPU Op", width=100),
            fill("XPU Time(ms)", width=30),
        ]
    )
    for i in range(max_len):
        if i < len(lhs_op_list) and i < len(rhs_op_list):
            lhs_op = lhs_op_list[i]
            rhs_op = rhs_op_list[i]
            table.add_row(
                [
                    fill(lhs.get_name(), width=200),
                    lhs_op.get_name(),
                    lhs_op.get_time(),
                    rhs_op.get_name(),
                    rhs_op.get_time(),
                ]
            )
        elif i < len(lhs_op_list):
            lhs_op = lhs_op_list[i]
            table.add_row(
                [fill("", width=200), lhs_op.get_name(), lhs_op.get_time(), "", ""]
            )
        elif i < len(rhs_op_list):
            rhs_op = rhs_op_list[i]
            table.add_row(
                [fill("", width=200), "", "", rhs_op.get_name(), rhs_op.get_time()]
            )
    table.add_row(
        [fill("", width=200), "GPU Total", lhs.get_time(), "XPU Total", rhs.get_time()]
    )
    return table


def compare(analyzer1, analyzer2):
    """
    Function:
    1. compare the two analyzers and return a table with the following format:
    """
    analyzer1.analysis()
    lhs_m_num = count_module(analyzer1.op_or_module)
    analyzer2.analysis()
    rhs_m_num = count_module(analyzer2.op_or_module)
    if rhs_m_num != lhs_m_num:
        Logger.error("The number of modules is not the same")

    lhs_modules = analyzer1.get_modules()
    rhs_modules = analyzer2.get_modules()
    lhs_block_list = []
    rhs_block_list = []
    for lhs_module, rhs_module in zip(lhs_modules, rhs_modules):
        lhs_sub_list, rhs_sub_list = compare_module(lhs_module, rhs_module)
        lhs_block_list.extend(lhs_sub_list)
        rhs_block_list.extend(rhs_sub_list)
    if len(lhs_block_list) != len(rhs_block_list):
        Logger.error("The number of blocks is not the same")
    return lhs_block_list, rhs_block_list


def gen_module_compare_tables(analyzer1, analyzer2):
    """
    Function:
    """
    lhs_block_list, rhs_block_list = compare(analyzer1, analyzer2)
    table_list = []
    for lhs_block, rhs_block in zip(lhs_block_list, rhs_block_list):
        if len(lhs_block.get_op_list()) > 0 or len(rhs_block.get_op_list()) > 0:
            table_list.append(merge_block(lhs_block, rhs_block))
    return table_list


def gen_module_compare_table_str(analyzer1, analyzer2):
    if isinstance(analyzer1, AtenOpAnalyzer) and isinstance(analyzer2, AtenOpAnalyzer):
        table_list = gen_module_compare_tables(analyzer1, analyzer2)
        for table in table_list:
            table_str += table.get_string() + "\n"
        return table_str
    elif isinstance(analyzer1, DistAnalyzer) and isinstance(analyzer2, DistAnalyzer):
        Logger.error("for distribution op, not supported yet.")
    else:
        Logger.error("The 2 analyzers are not the same type")
