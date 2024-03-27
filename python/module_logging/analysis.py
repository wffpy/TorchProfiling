from lzma import FORMAT_ALONE
import os.path as osp
import re
from xml.etree.ElementTree import C14NWriterTarget
import prettytable as pt
from .logging import Logger
from enum import Enum, auto
from textwrap import fill


class STATE(Enum):
    BEGIN = auto()
    MODULE = auto()
    FORMAL = auto()
    STOP = auto()

class LocalModule(object):
    def __init__(self, name:str = "") -> None:
        self._name_ = name
        self.element_list = []

    def get_name(self):
        return self._name_
    def add_elem(self, elem):
        self.element_list.append(elem)
    
    def get_elem_list(self):
        return self.element_list
    

class LocalOp(object):
    def __init__(self, name = "", time = 0) -> None:
        self._name_ = name
        self._time_ = time
    def set_time(self, time):
        self._time_ += time

    def get_name(self):
        return self._name_

    def get_time(self):
        return self._time_

class OpSummary(object):
    def __init__(self, time) -> None:
        self.call_count = 1
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
    

class ModuleStack(object):
    def __init__(self) -> None:
        self.module_list = []
    
    def push(self, module):
        self.module_list.append(module)
    
    def pop(self):
        self.module_list.pop(0)
    
    def depth(self):
        return len(self.module_list)

    def top(self):
        return self.module_list[0]


class Analyzer:
    def __init__(self, path):
        '''
        args:
            path: the path of log file
        '''
        if not osp.exists(path):
            raise FileNotFoundError("log file {} doesn't find".format(path))
        self.log_path = path
        # current module name
        self.current_m_name = ""
        self.collection_state = STATE.BEGIN
        self.current_op_name = ""
        self.current_op = None
        self.current_module = None 
        self.stack = ModuleStack()
        self.op_or_module = []
        self.total = 0

    def identify_step_beign_or_end(self, line:str):
        '''
        args:
            line: a string of log file
        return:
            bool
        '''
        if "iteration" in line and "learning" in line and "loss" in line:
            if self.collection_state == STATE.BEGIN:
                Logger.debug("Step Begin")
                self.collection_state = STATE.FORMAL
            else:
                Logger.debug("Step End")
                self.collection_state = STATE.STOP 
            return True
        return False

    
    def identify_module_begin(self, line:str):
        '''
        args:
            line: a string of log file
        return:
            bool
        Function:
        1. identify the module by following symbols: [BEGIN FORWARD], [END FORWARD], [BEGIN BACKWARD], [END BACKWARD]o
        2. if there is no module in this iteration, return None
        '''
        if (self.collection_state == STATE.FORMAL or self.collection_state == STATE.MODULE) and line.startswith("[BEGIN FORWARD]") or line.startswith("[BEGIN BACKWARD]"):
            Logger.debug("Module Begin")
            if self.current_module is not None:
                self.stack.push(self.current_module)
            self.current_m_name = line.rstrip("\n").split(":")[-1]
            self.current_module = LocalModule(self.current_m_name)
            self.collection_state = STATE.MODULE
            return True
        return False
    
    def identify_module_end(self, line:str):
        '''
        args:
            line: a string of log file
        return:
            module_name: the name of module
        Function:
        1. identify the module by following symbols: [BEGIN FORWARD], [END FORWARD], [BEGIN BACKWARD], [END BACKWARD]o
        2. if there is no module in this iteration, return None
        '''
        if self.collection_state == STATE.MODULE and (line.startswith("[END FORWARD]") or line.startswith("[END BACKWARD]")):
            Logger.debug("Module End")
            temp_module = self.current_module
            self.current_module = self.stack.top() if 0 < self.stack.depth() else None
            self.current_m_name = self.current_module.get_name() if self.current_module  else ""
            self.collection_state = STATE.FORMAL if self.current_module is None else STATE.MODULE
            if self.current_module is None:
                self.op_or_module.append(temp_module)
            else:
                self.current_module.add_elem(temp_module)
            if 0 < self.stack.depth():
                self.stack.pop()

            return True
        return False
    
    def identify_op_start(self, line:str):
        '''
        args:
            line: a string of log file
        return:
            bool
        Function:
        1. identify the op by following symbols: [START_SYMBOL], [END_SYMBOL]
        2. if there is no op in this iteration, return None
        '''
        if (self.collection_state == STATE.FORMAL or self.collection_state == STATE.MODULE) and line.startswith("[START_SYMBOL]"):
            Logger.debug("Op Start")
            self.current_op_name = line.rstrip("\n").split(":")[-1].replace("_", " ")
            self.current_op = LocalOp(self.current_op_name)
            return True
        return False

    def identify_op_end(self, line:str):
        if (self.collection_state == STATE.FORMAL or self.collection_state == STATE.MODULE) and line.startswith("[END_SYMBOL]"):
            Logger.debug("Op End")
            self.total += self.current_op.get_time()
            if self.current_module is not None:
                self.current_module.add_elem(self.current_op)
            else:
                self.op_or_module.append(self.current_op)
            return True
        return False
    
    def identify_op_time(self, line:str):
        if (self.collection_state == STATE.FORMAL or self.collection_state == STATE.MODULE) and line.startswith("[XPURT_PROF]"):
            Logger.debug("Op Time")
            self.current_op.set_time(float(line.split(" ")[-2])/1000000)
            return True
        return False
    
    def get_total(self):
        return self.total
            
    def analysis(self):
        lines = []
        with open(self.log_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            self.identify_step_beign_or_end(line)
            if self.collection_state == STATE.BEGIN:
                continue
            elif self.collection_state == STATE.STOP:
                break
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
        '''
        Function:
        1. generate a table with the following format:
        Module                     Aten Op               Time(ms)
        ----------------------------------------------------------
        Conv2d                     conv2d                10.3456789
        BatchNorm2d                 batch_norm            1.2345678
        '''
        _list = []
        final_list = []
        for elem in self.op_or_module:
            if isinstance(elem, LocalModule):
                t = (elem.get_name(), elem)
                _list.append(t)
            elif isinstance(elem, LocalOp):
                t = ("", elem)
                _list.append(t) 
            else:
                Logger.error("elem is not LocalModule or LocalOp")

            while len(_list) > 0:
                local_elem = _list.pop(0)
                if isinstance(local_elem[1], LocalModule):
                    for m_elem in local_elem[1].get_elem_list():
                        if isinstance(m_elem, LocalModule):
                            _list.append((m_elem.get_name(), m_elem))
                        elif isinstance(m_elem, LocalOp):
                            _list.append((local_elem[0], m_elem)) 
                elif isinstance(local_elem[1], LocalOp):
                    final_list.append(local_elem) 
        
        table = pt.PrettyTable(['Module', 'Aten Op', 'Time'])
        for elem in final_list:
            table.add_row([fill(elem[0], width=50), elem[1].get_name(), elem[1].get_time()])
        table.set_style(pt.DEFAULT)
        table.align = "l"
        return table

    def gen_max_min_avg_table(self):
        '''
        Function:
        1. generate a summary table with the following format:
        Op                          Max Time(ms)   Min Time(ms)    Avg Time(ms)    Total Time(ms)  Count
        --------------------------------------------------------------------------------------------
        conv2d                      10.3456789     1.2345678       1.2345678         123456789   1000
        '''
        _list = []
        final_list = []
        for elem in self.op_or_module:
            _list.append(elem)
            while len(_list) > 0:
                local_elem = _list.pop(0)
                if isinstance(local_elem, LocalModule):
                    for m_elem in local_elem.get_elem_list():
                        _list.append(m_elem)
                elif isinstance(local_elem, LocalOp):
                    final_list.append(local_elem) 
                else:
                    Logger.error("elem is not LocalModule or LocalOp")
        
        op_dict = {} 
        for elem in final_list:
            k = elem.get_name()
            v = elem.get_time()
            if k in op_dict:
                op_dict[k].add_time(v)
            else:
                op_summary = OpSummary(v)
                op_dict[k] = op_summary
        
        op_list = sorted(op_dict.items(), key=lambda x: x[1].get_total_time(), reverse=True)

        table = pt.PrettyTable(['Op', 'Max Time(ms)', 'Min Time(ms)', 'Avg Time(ms)', "Total Time(ms)", "Count", "Percent(%)"])
        for op in op_list:
            percent = op[1].get_total_time() / self.get_total()
            table.add_row([fill(op[0], width=40), op[1].get_max(), op[1].get_min(), op[1].get_avg(), op[1].get_total_time(), op[1].get_call_count(), percent])
        table.align = "l"
        return table
    
    def gen_total_time_table(self):
        '''
        Function:
        1. generate a summary table with the following format:
        Total Time(ms)
        --------------------------------------------------
        10000.123456789
        '''
        table = pt.PrettyTable(["Total Time(ms)"])
        table.add_row([self.get_total()])
        return table
        
