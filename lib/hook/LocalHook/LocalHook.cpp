#include "hook/LocalHook/LocalHook.h"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <link.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#include "capstone/capstone.h"
#include "capstone/x86.h"

#include "hook/CFuncHook.h"
#include "utils/Log/Log.h"
#include "utils/Utils.h"

namespace local_hook {

#ifdef __x86_64__

#define RELATIVE_JUMP_INST_SIZE 5
#define ABSOLUTE_JUMP_INST_SIZE 13
#define ENDBR_INST_SIZE 4

#define ENDBR_INST 0xfa1e0ff3
#define NOP_INST 0x90

#endif

uintptr_t find_free_address(uintptr_t aligned_addr, size_t size) {
    pid_t pid = getpid();
    std::string maps_path = "/proc/" + std::to_string(pid) + "/maps";
    std::ifstream maps_file(maps_path.c_str());
    std::string line;
    uintptr_t start = 0;
    uintptr_t end = 0;
    uintptr_t result = aligned_addr;
    while (std::getline(maps_file, line)) {
        std::istringstream iss(line);
        iss >> std::hex >> start;
        iss.ignore(std::numeric_limits<std::streamsize>::max(), '-');
        iss >> std::hex >> end;
        if (end < aligned_addr) {
            continue;
        }
        if (result < start && result + size < start) {
            break;
        } else {
            result = end;
        }
    }
    return result;
}

// void *alloc_page_near_address(void *target_addr) {
//     static uint64_t page_size = getpagesize();
//     uintptr_t aligned_addr = ((uintptr_t)target_addr) & (~(page_size - 1));
//     uintptr_t free_mem = find_free_address((uintptr_t)aligned_addr,
//     page_size); void *mmap_addr = mmap((void *)free_mem, page_size, PROT_READ
//     | PROT_EXEC,
//                            MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
//     if (mmap_addr == MAP_FAILED) {
//         perror("mmap");
//         std::cout << "mmap failed" << std::endl;
//         exit(0);
//     }
//     std::cout << "mmap success: " << mmap_addr << std::endl;
//     return mmap_addr;
// }

struct X64Instructions {
    cs_insn *instructions;
    uint32_t numInstructions;
    uint32_t numBytes;
};

// int64_t check_func_mem(void *function) {
//     csh handle;
//     auto s = cs_open(CS_ARCH_X86, CS_MODE_64, &handle);
//     if (s != 0) {
//         std::cout << "Error opening capstone handle" << std::endl;
//     }
//     s = cs_option(handle, CS_OPT_DETAIL,
//                   CS_OPT_ON); // we need details enabled for relocating RIP
//                               // relative instrs
//     if (s != 0) {
//         std::cout << "Error set option" << std::endl;
//     }

//     uint32_t byte_count = 0;

//     // for cpu with BIT check, the first instruction of a function is endbr
//     // endbr64 instruction: 0xfa1e0ff3
//     uint32_t endbr64 = 0xfa1e0ff3;
//     uint8_t *code = (uint8_t *)function;
//     if (endbr64 == *(uint32_t *)function) {
//         code = (uint8_t *)function + 4;
//         byte_count += 4;
//     }

//     cs_insn *disassembled_instrs = nullptr;
//     size_t count =
//         cs_disasm(handle, code, 50, (uint64_t)code, 0, &disassembled_instrs);
//     if (count == 0) {
//         s = cs_errno(handle);
//         std::cout << "error status: " << cs_strerror(s) << std::endl;
//     }

//     for (int32_t i = 0; i < count; ++i) {
//         cs_insn &inst = disassembled_instrs[i];
//         byte_count += inst.size;
//     }

//     cs_free(disassembled_instrs, count);
//     cs_close(&handle);
//     return byte_count;
// }

X64Instructions steal_bytes(void *function, int64_t bytes) {
    // Disassemble stolen bytes
    std::cout << "Stealing bytes from: " << function << std::endl;
    csh handle;
    auto s = cs_open(CS_ARCH_X86, CS_MODE_64, &handle);
    if (s != 0) {
        ELOG() << "error opening capstone handle";
    }
    // we need details enabled for relocating RIP relative instrs
    s = cs_option(handle, CS_OPT_DETAIL, CS_OPT_ON); 
    CHECK(s == 0, "error set option");

    // for cpu with BIT check, the first instruction of a function is endbr
    // endbr64 instruction: 0xfa1e0ff3
    uint32_t endbr = ENDBR_INST;
    uint8_t *code = (uint8_t *)function;
    if (endbr == *(uint32_t *)function) {
        code = (uint8_t *)function + 4;
    }

    cs_insn *disassembled_instrs = nullptr;
    size_t count = cs_disasm(handle, code, 20 + bytes, (uint64_t)code,
                             20 + bytes, &disassembled_instrs);
    if (count == 0) {
        s = cs_errno(handle);
        ELOG() << "error status: " << cs_strerror(s);
    }

    // get the instructions covered by the first 9 bytes of the original
    // function
    uint32_t byte_count = 0;
    uint32_t instr_count = 0;
    for (int32_t i = 0; i < count; ++i) {
        cs_insn &inst = disassembled_instrs[i];
        byte_count += inst.size;
        instr_count++;
        if (byte_count >= bytes)
            break;
    }

    // replace instructions in target func wtih NOPs
    uint8_t nop = NOP_INST; 
    memset((void *)code, nop, byte_count);

    cs_close(&handle);
    return {disassembled_instrs, instr_count, byte_count};
}

void enable_mem_write(void *func) {
    static int64_t page_size = getpagesize();

    // get the start address of the page containing the function
    uintptr_t page_start = ((uintptr_t)func) & (~(page_size - 1));

    // set the memory protections to read/write/execute, so that we can modify
    // the instructions
    if (mprotect((void *)page_start, page_size,
                 PROT_READ | PROT_WRITE | PROT_EXEC) == -1) {
        perror("mprotect");
    }
}

struct MemInfo {
    void *addr;
    int64_t size;
};

class MemAllocator {
  public:
    MemAllocator() {}
    ~MemAllocator();
    void *alloc(void *target_addr, int64_t size = 1024);

  private:
    void *alloc_page_near_address(void *target_addr, int64_t size);
    std::vector<MemInfo> mem_blocks;
};

void *MemAllocator::alloc_page_near_address(void *target_addr,
                                            int64_t aligned_size) {
    static uint64_t page_size = getpagesize();
    uintptr_t aligned_addr = ((uintptr_t)target_addr) & (~(page_size - 1));
    uintptr_t free_mem = find_free_address((uintptr_t)aligned_addr, page_size);
    void *mmap_addr =
        mmap((void *)free_mem, aligned_size, PROT_READ | PROT_EXEC,
             MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mmap_addr == MAP_FAILED) {
        perror("mmap");
        CHECK(false, "mmap failed in alloc_page_near_address function");
    }
    return mmap_addr;
}

void *MemAllocator::alloc(void *target_addr, int64_t size) {
    static uint64_t page_size = getpagesize();
    DLOG() << "memory page size: " << page_size;
    int64_t page_num =
        size % page_size == 0 ? size / page_size : 1 + size / page_size;
    DLOG() << "page num: " << page_num;
    int64_t aligned_size = page_num * page_size;
    DLOG() << "real alloc size: " << aligned_size;
    void *addr = alloc_page_near_address(target_addr, aligned_size);
    mem_blocks.emplace_back(MemInfo{addr, aligned_size});
    return addr;
}

MemAllocator::~MemAllocator() {
    for (auto &mem_info : mem_blocks) {
        munmap(mem_info.addr, mem_info.size);
    }
}

typedef utils::Singleton<MemAllocator> MemAllocatorSingleton;

class HookImpl {
  public:
    HookImpl() {}
    static bool has_endbr(void *func_ptr);
    // static bool is_inst(cs_insn &inst, );
    static bool is_rip_relative_inst(cs_insn &inst);
    static bool is_jump(cs_insn &inst);
    void build();

  private:
    void *payload_func;
    void *hook_func;
    // used to store the trampoline func and relay func
    void *hook_mem;
};

// check if the first instruction is endbr64
bool HookImpl::has_endbr(void *func_ptr) {
    uint32_t endbr = ENDBR_INST;
    return *(uint32_t *)func_ptr == endbr;
}

// check if the instruction is a RIP relative address
bool HookImpl::is_rip_relative_inst(cs_insn &inst) {
    cs_x86 *x86 = &(inst.detail->x86);
    for (uint32_t i = 0; i < x86->op_count; ++i) {
        cs_x86_op *op = &(x86->operands[i]);
        if (op->type == X86_OP_MEM) {
            return op->mem.base == X86_REG_RIP;
        }
    }
    return false;
}

// check if the instruction is a jump
bool HookImpl::is_jump(cs_insn &inst) {
    if (inst.id >= X86_INS_JAE && inst.id <= X86_INS_JS) {
        return true;
    }
    return false;
}

void HookImpl::build() {
    // step1: build Trampoline Func
    // step2: build Relay Func
    // step3: rewrite Hooked Func
}

void write_absolute_jump64(void *relay_func_mem, void *jmp_target) {
    // uint8_t abs_jmp_instrs[] = {0xf3, 0x0f, 0x1e, 0xfa, 0x49, 0xBA,
    uint8_t abs_jmp_instrs[] = {0x49, 0xBA, 0x00, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x41, 0xFF, 0xE2};

    uintptr_t jump_target_addr = (uintptr_t)jmp_target;
    memcpy(&abs_jmp_instrs[2], &jump_target_addr, sizeof(jump_target_addr));
    memcpy(relay_func_mem, abs_jmp_instrs, sizeof(abs_jmp_instrs));
}

/// @brief
/// @param write_addr
/// @param call_target
/// @return: the bytes of write instructions
uint32_t write_absolute_call64(void *write_addr, void *call_target) {
    uint8_t abs_call_instrs[] = {// 0xf3, 0x0f, 0x1e, 0xfa,
                                 0x49, 0xBA, 0x00, 0x00, 0x00, 0x00, 0x00,
                                 0x00, 0x00, 0x00, 0x41, 0xFF, 0xD2};

    uint64_t call_target_addr = (uint64_t)call_target;
    memcpy(&abs_call_instrs[2], &call_target_addr, sizeof(call_target_addr));
    memcpy(write_addr, abs_call_instrs, sizeof(abs_call_instrs));
    // return 17;
    return 13;
}

// check if the instruction operand is a RIP relative address
bool is_rip_relative_inst(cs_insn &inst) {
    cs_x86 *x86 = &(inst.detail->x86);
    for (uint32_t i = 0; i < x86->op_count; ++i) {
        cs_x86_op *op = &(x86->operands[i]);
        if (op->type == X86_OP_MEM) {
            return op->mem.base == X86_REG_RIP;
        }
    }
    return false;
}

bool check_mem_offset(int64_t offset, int64_t bytes) {
    DLOG() << "offset: " << offset << ", bytes: " << bytes;
    switch (bytes) {
    case 1: {
        return INT8_MIN < offset && offset < INT8_MAX;
    }
    case 2: {
        return INT16_MIN < offset && offset < INT16_MAX;
    }
    case 4: {
        return INT32_MIN < offset && offset < INT32_MAX;
    }
    case 8: {
        return INT64_MIN < offset && offset < INT64_MAX;
    }
    default: {
        ELOG() << "Unsupported operand size: " << bytes;
        return false;
    }
    }
    return false;
}

/// @brief instruction has been move to a new address,
/// so need to update the operand which is relative address.
/// @param inst
/// @param new_addr
bool relocate_instruction(cs_insn &inst, void *new_addr) {
    cs_x86 *x86 = &(inst.detail->x86);
    uint64_t inst_addr = inst.address;
    // 1. traverse the operands of instruction, find the one that is relative
    // address
    for (uint32_t i = 0; i < x86->op_count; ++i) {
        cs_x86_op *op = &(x86->operands[i]);
        if (op->type == X86_OP_MEM && op->mem.base == X86_REG_RIP) {
            uint8_t bytes = op->size;
            int64_t disp = op->mem.disp;
            std::cout << "origin disp: " << std::hex << disp << std::endl;
            disp -= (int64_t)((uint64_t)new_addr - inst_addr);
            std::cout << "new dis: " << std::hex << disp << std::endl;
            std::cout << "bytes: " << std::dec << bytes << std::endl;
            if (!check_mem_offset(disp, 4)) {
                std::cout << "Invalid displacement: " << disp << std::endl;
                return false;
            }
            op->mem.disp = disp;
        }
    }
    return true;
}

// template<class T>
// T GetDisplacement(cs_insn* inst, uint8_t offset)
// {
//     T disp;
//     memcpy(&disp, &inst->bytes[offset], sizeof(T));
//     return disp;
// }

// //rewrite instruction bytes so that any RIP-relative displacement operands
// //make sense with wherever we're relocating to
// void RelocateInstruction(cs_insn* inst, void* dstLocation)
// {
//     cs_x86* x86 = &(inst->detail->x86);
//     uint8_t offset = x86->encoding.disp_offset;

//     uint64_t displacement = inst->bytes[x86->encoding.disp_offset];
//     switch (x86->encoding.disp_size)
//     {
//     case 1:
//     {
//         int8_t disp = GetDisplacement<uint8_t>(inst, offset);
//         disp -= int8_t(uint64_t(dstLocation) - inst->address);
//         memcpy(&inst->bytes[offset], &disp, 1);
//     }break;

//     case 2:
//     {
//         int16_t disp = GetDisplacement<uint16_t>(inst, offset);
//         disp -= int16_t(uint64_t(dstLocation) - inst->address);
//         memcpy(&inst->bytes[offset], &disp, 2);
//     }break;

//     case 4:
//     {
//         int32_t disp = GetDisplacement<int32_t>(inst, offset);
//         disp -= int32_t(uint64_t(dstLocation) - inst->address);
//         memcpy(&inst->bytes[offset], &disp, 4);
//     }break;
//     }
// }

bool is_jump(cs_insn &inst) {
    if (inst.id >= X86_INS_JAE && inst.id <= X86_INS_JS) {
        return true;
    }
    return false;
}

bool is_cmp(cs_insn &inst) {
    if (inst.id == X86_INS_CMP) {
        return true;
    }
    return false;
}

// 1. conditional jump instructions must be relative jump
// 2. uncondition jump: if jump with a relative address, the instruction must
// start with 0xeb or 0xe9
bool is_relative_jump(cs_insn &inst) {
    bool is_any_jump = inst.id >= X86_INS_JAE && inst.id <= X86_INS_JS;
    bool is_jmp = inst.id == X86_INS_JMP;
    bool start_with_eb_or_e9 = inst.bytes[0] == 0xeb || inst.bytes[0] == 0xe9;
    return is_jmp ? start_with_eb_or_e9 : is_any_jump;
}

bool is_relative_instr(cs_insn &inst, int64_t inst_type) {
    if (inst.id == inst_type) {
        return is_rip_relative_inst(inst);
    }
    return false;
}

uint32_t extend_jmp_to_abs_table(cs_insn &inst, uint8_t *write_addr) {
    if (!is_jump(inst)) {
        ELOG() << "not a jump instruction: " << inst.mnemonic;
    }

    uint64_t target_addr = strtoull(inst.op_str, NULL, 0);
    write_absolute_jump64((void *)write_addr, (void *)target_addr);

    // auto operand_type = inst.detail->x86.operands[0].type;
    // if (operand_type == X86_OP_IMM) {
    //     int64_t target_addr = inst.detail->x86.operands[0].imm;
    // } else {
    //     std::cout << "Unsupported operand type: " << operand_type <<
    //     std::endl; exit(0);
    // }
    return 13;
}

/// @brief
/// @param inst origin instruction
/// @param write_addr the address to store the jmp instruction
/// @param target_addr the target address of the jmp instruction
/// @return
void rewrite_jmp_instruction(cs_insn &inst, uint8_t *write_addr,
                             uint8_t *target_addr) {
    uint64_t jmp_offset = target_addr - (write_addr + inst.size);

    int64_t operand_size = 0;
    if (inst.bytes[0] == 0x0f) {
        // jmp instruction starts with 0x0f, op code is 2 bytes
        operand_size = inst.size - 2;
    } else {
        operand_size = inst.size - 1;
    }

    // check jmp offset
    if (!check_mem_offset(jmp_offset, operand_size)) {
        ELOG() << "Invalid jmp offset: " << jmp_offset;
    }

    if (1 == operand_size) {
        inst.bytes[operand_size] = jmp_offset;
    } else if (2 == operand_size) {
        uint16_t jmp_offset16 = jmp_offset;
        memcpy(&inst.bytes[operand_size], &jmp_offset16, operand_size);
    } else if (4 == operand_size) {
        uint32_t jmp_offset32 = jmp_offset;
        memcpy(&inst.bytes[operand_size], &jmp_offset32, operand_size);
    }
}

int64_t extend_jmp_with_address_to_abs_table(cs_insn &inst, uint8_t *write_addr) {
    cs_x86 *x86 = &(inst.detail->x86);
    int64_t inst_size = inst.size;
    uint64_t inst_addr = inst.address;
    // move: 0x49, 0xBA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    // 41 ff 22 : jmp    QWORD PTR [r10]
    uint8_t abs_ptr_jmp_instrs[] = {0x49, 0xBA, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x41, 0xFF, 0x22};

    int64_t new_insts_size = sizeof(abs_ptr_jmp_instrs);
    int64_t operand_cnt = x86->op_count;
    CHECK(operand_cnt == 1, "the operand for jmp is not 1");
    cs_x86_op *op = &(x86->operands[0]);
    CHECK(op->type == X86_OP_MEM && op->mem.base == X86_REG_RIP,
          "the instruction operand is not relative memory address");
    int64_t disp = op->mem.disp;
    uint64_t abs_addr = inst_addr + inst_size + disp;
    DLOG() << "abs_addr: " << std::hex << abs_addr;
    memcpy(&abs_ptr_jmp_instrs[2], &abs_addr, sizeof(abs_addr));
    memcpy(write_addr, abs_ptr_jmp_instrs, new_insts_size);
    return new_insts_size;
}

// rewrite the jump which use register or an address to store the destination
// address
void rewrite_jmp_with_address(cs_insn &inst, uint8_t *write_addr,
                              uint8_t *target_addr) {
    uint8_t jmp_bytes[2] = {0xEB, 0x0};
    int64_t jmp_offset = target_addr - (write_addr + sizeof(jmp_bytes));
    if (jmp_offset > INT8_MAX || jmp_offset < INT8_MIN) {
        ELOG() << "Invalid jmp offset(not in the range of INT8) : " << jmp_offset;
    }
    uint8_t u8_jmp_offset = jmp_offset;
    memcpy(&jmp_bytes[1], &u8_jmp_offset, sizeof(jmp_bytes) - 1);

    uint8_t nop = NOP_INST;
    memset(inst.bytes, nop, inst.size);
    memcpy(inst.bytes, jmp_bytes, sizeof(jmp_bytes));
}

uint32_t extend_call_to_abs_table(cs_insn &inst, uint8_t *write_addr,
                               uint8_t *jump_back_addr) {
    uint64_t target_addr = strtoull(inst.op_str, NULL, 0);
    uint32_t written_bytes =
        write_absolute_call64((void *)write_addr, (void *)target_addr);
    write_addr += written_bytes;
    // add jmp to the final jmp instruction which jump to the current call
    // instruction uint8_t jmp_bytes[2] = { 0xEB, uint8_t(jump_back_addr -
    // (write_addr + sizeof(jmp_bytes))) };
    uint8_t jmp_bytes[2] = {
        0xEB, uint8_t(jump_back_addr - (write_addr + sizeof(jmp_bytes)))};
    memcpy(write_addr, jmp_bytes, sizeof(jmp_bytes));

    return written_bytes + sizeof(jmp_bytes);
}

void rewrite_call_instruction(cs_insn &inst, uint8_t *write_addr,
                              uint8_t *target_addr) {
    int64_t jmp_offset = target_addr - (write_addr + inst.size);
    if (jmp_offset > INT8_MAX || jmp_offset < INT8_MIN) {
        ELOG() << "Invalid jmp offset inner trampoline: " << jmp_offset;
    }
    // this is jump inner trampoline function,
    // represent the offset of jmp instruction with 8-bit
    uint8_t u8_jmp_offset = jmp_offset;
    // construct jmp instruction
    uint8_t jmp_bytes[2] = {0xEB, u8_jmp_offset};
    uint8_t nop = NOP_INST;
    memset(inst.bytes, nop, inst.size);
    memcpy(inst.bytes, jmp_bytes, sizeof(jmp_bytes));
}

void rewrite_cmpl_instruction(cs_insn &inst, uint8_t *write_addr,
                              uint8_t *target_addr) {
    int64_t jmp_offset = target_addr - (write_addr + 2);
    if (jmp_offset > INT8_MAX || jmp_offset < INT8_MIN) {
        ELOG() << "Invalid jmp offset inner trampoline: " << jmp_offset;
    }
    // this is jump inner trampoline function,
    // represent the offset of jmp instruction with 8-bit
    uint8_t u8_jmp_offset = jmp_offset;
    // construct jmp instruction
    uint8_t jmp_bytes[2] = {0xEB, u8_jmp_offset};
    uint8_t nop = NOP_INST;
    memset(inst.bytes, nop, inst.size);
    memcpy(inst.bytes, jmp_bytes, sizeof(jmp_bytes));
}

// stor the lhs/rhs into r10/r11, then use 64-bit compare instruction
int64_t gen_cmpl_with_register(uint64_t lhs, uint64_t rhs, void *store_mem) {
    // mov r10, 0x0000000000000000
    uint8_t mov_r10_instr[] = {0x49, 0xBA, 0x00, 0x00, 0x00,
                               0x00, 0x00, 0x00, 0x00, 0x00};
    // mov r11, 0x0000000000000000
    // uint8_t mov_r9_instr[] = {0x49, 0xB9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // 0x00, 0x00};
    uint8_t mov_r9_instr[] = {0x49, 0xBB, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00, 0x00};
    // cmp r10, r11
    uint8_t cmp_instr[] = {0x4D, 0x39, 0xD1};

    memcpy(&mov_r9_instr[2], &lhs, sizeof(uint64_t));
    memcpy(&mov_r10_instr[2], &rhs, sizeof(uint64_t));

    int64_t mov_r9_bytes = sizeof(mov_r9_instr);
    int64_t mov_r10_bytes = sizeof(mov_r10_instr);
    int64_t cmp_bytes = sizeof(cmp_instr);

    memcpy(store_mem, mov_r9_instr, mov_r9_bytes);
    memcpy(store_mem + mov_r9_bytes, mov_r10_instr, mov_r10_bytes);
    memcpy(store_mem + mov_r9_bytes + mov_r10_bytes, cmp_instr, cmp_bytes);
    return mov_r9_bytes + mov_r10_bytes + cmp_bytes;
}

int64_t get_imme_from_operand(cs_insn &instr, cs_x86_op &operand) {
    int64_t val = 0;
    x86_op_type type = operand.type;
    if (type == X86_OP_IMM) {
        val = operand.imm;
    } else if (type == X86_OP_MEM) {
        x86_op_mem mem = operand.mem;
        if (mem.base == X86_REG_RIP) {
            uint64_t address = instr.address;
            uint64_t instr_size = instr.size;
            uint64_t rip = address + instr_size;
            int64_t mem_disp = mem.disp;
            val = mem_disp + rip;
        } else {
            val = mem.disp;
        }
    } else {
        CHECK(false, "not support operand type");
    }
    return val;
}


// replace the origin relative compare instruction with
// absolute 64-bit compare instructions, then jump back to the
// the end of origin compare instruction.
int64_t add_cmp_to_abs_table(cs_insn &inst, void *write_addr,
                             void *jump_back_addr) {
    cs_x86 x86_instr = inst.detail->x86;
    int64_t operand_num = x86_instr.op_count;
    if (operand_num != 2) {
        CHECK(false, "cmp op has more than two operands");
    }

    cs_x86_op lhs_operand = x86_instr.operands[0];
    cs_x86_op rhs_operand = x86_instr.operands[1];
    int64_t lhs_val = get_imme_from_operand(inst, lhs_operand);
    int64_t rhs_val = get_imme_from_operand(inst, rhs_operand);

    int64_t cmpl_bytes = gen_cmpl_with_register(lhs_val, rhs_val, write_addr);
    DLOG() << "extended compare instrustions bytes:" << cmpl_bytes;

    // jump back to the origin instruction
    uint8_t jmp_instr[2] = {0xEB, uint8_t((uint8_t *)jump_back_addr -
                                          ((uint8_t *)write_addr + cmpl_bytes +
                                           sizeof(jmp_instr)))};
    memcpy(write_addr + cmpl_bytes, jmp_instr, sizeof(jmp_instr));
    return cmpl_bytes + sizeof(jmp_instr);
}

/**
 *       __________________________________
 *      |______________......._____________|------|
 *      |______________......._____________|      |--> stolen instructions
 *      |______________......._____________|------|
 *      |__________mov r10, addr___________| 10 bytes = 2bytes + 8 bytes
 *      |__________jmpq r10________________| 3 bytes
 *      |__________________________________|
 **/
int64_t build_callback(void *func2hook, void *hook_mem, int64_t bytes,
                       bool endbr = true) {
    X64Instructions stolenInstrs = steal_bytes(func2hook, bytes);
    WLOG() << "stolen bytes: " << stolenInstrs.numBytes;
    WLOG() << "stolen instruction number: " << stolenInstrs.numInstructions;

    uint64_t endbr_bytes = endbr ? ENDBR_INST_SIZE : 0;
    uint64_t mov_bytes = 10;
    uint64_t jmp_abs_bytes = 3;
    uint8_t *stolen_byte_mem = (uint8_t *)hook_mem;
    uint8_t *jumpBackMem =
        stolen_byte_mem + endbr_bytes + stolenInstrs.numBytes;
    // absolute table:
    uint8_t *abs_table_mem = jumpBackMem + mov_bytes + jmp_abs_bytes;
    DLOG() << "absolute table start: " << std::hex << (uintptr_t)abs_table_mem;

    if (endbr) {
        uint8_t endbr_instr[4] = {0xf3, 0x0f, 0x1e, 0xfa};
        memcpy(hook_mem, endbr_instr, sizeof(endbr_instr));
        stolen_byte_mem += sizeof(endbr_instr);
    }

    for (uint32_t i = 0; i < stolenInstrs.numInstructions; ++i) {
        cs_insn &inst = stolenInstrs.instructions[i];
        if (inst.id >= X86_INS_LOOP && inst.id <= X86_INS_LOOPNE) {
            // bail out on loop instructions, I don't have a good way
            // of handling them
            ELOG() << "loop instruction: " << inst.id;
            return 0;
        }
        if (is_cmp(inst) && is_rip_relative_inst(inst)) {
            WLOG() << "relative compare instruction: " << inst.id;
            uint8_t *jump_back_addr = stolen_byte_mem + inst.size;
            int64_t abs_cmp_size = add_cmp_to_abs_table(
                inst, (void *)abs_table_mem, (void *)jump_back_addr);
            rewrite_cmpl_instruction(inst, stolen_byte_mem, abs_table_mem);
            abs_table_mem += abs_cmp_size;
        } else if (is_relative_jump(inst)) {
            WLOG() << "relative jump instruction: " << inst.id;
            uint64_t abs_jmp_size = extend_jmp_to_abs_table(inst, abs_table_mem);
            rewrite_jmp_instruction(inst, stolen_byte_mem, abs_table_mem);
            abs_table_mem += abs_jmp_size;
        } else if (is_relative_instr(inst, X86_INS_JMP)) {
            WLOG() << "64 bits relative jump instruction: " << inst.id;
            uint64_t abs_jmp_size =
                extend_jmp_with_address_to_abs_table(inst, abs_table_mem);
            rewrite_jmp_with_address(inst, stolen_byte_mem, abs_table_mem);
            abs_table_mem += abs_jmp_size;
        } else if (inst.id == X86_INS_CALL) {
            WLOG() << "call instruction: " << inst.id;
            uint8_t *jump_back_addr = stolen_byte_mem + inst.size;
            uint32_t abs_call_size =
                extend_call_to_abs_table(inst, abs_table_mem, jump_back_addr);
            rewrite_call_instruction(inst, stolen_byte_mem, abs_table_mem);
            abs_table_mem += abs_call_size;
        } else if (is_rip_relative_inst(inst)) {
            ELOG() << "not handled rip relative instruction: " << inst.id;
            relocate_instruction(inst, stolen_byte_mem);
        }
        memcpy(stolen_byte_mem, inst.bytes, inst.size);
        stolen_byte_mem += inst.size;
    }

    write_absolute_jump64(jumpBackMem, (uint8_t *)func2hook + endbr_bytes +
                                           stolenInstrs.numBytes);
    free(stolenInstrs.instructions);
    return abs_table_mem - (uint8_t *)hook_mem;
}

bool has_endbr(void *func_ptr) {
    uint32_t endbr = ENDBR_INST;
    if (endbr == *(uint32_t *)func_ptr) {
        return true;
    }
    return false;
}

void install_local_hook(void *hooked_func, void *payload_func,
                        void **trampoline_ptr) {
    // as the payload function and hooked function maybe compiled with different
    // if they have endbr64 bytes is not unified.
    bool endbr_hooked = has_endbr(hooked_func);
    bool endbr_payload = has_endbr(payload_func);
    int64_t endbr_hooked_bytes = endbr_hooked ? ENDBR_INST_SIZE : 0;
    int64_t endbr_payload_bytes = endbr_payload ? ENDBR_INST_SIZE : 0;

    uint8_t jmp_instrs[5] = {0xE9, 0x0, 0x0, 0x0, 0x0};
    enable_mem_write(hooked_func);

    // TODO(may have bug):
    int64_t addr_distance = ((uint64_t)payload_func + endbr_payload_bytes) -
                            ((uint64_t)hooked_func + endbr_hooked_bytes + 5);

    // alloc memory for stolen bytes and relay function
    void *hook_mem =
        MemAllocatorSingleton::instance().get_elem()->alloc(hooked_func);
    DLOG() << "hook_mem: " << std::hex << hook_mem;

    int64_t trampoline_distance = (uint64_t)hook_mem - (uint64_t)hooked_func;
    int64_t try_stolen_bytes =
        trampoline_distance < INT32_MAX && trampoline_distance > INT32_MIN
            ? RELATIVE_JUMP_INST_SIZE
            : ABSOLUTE_JUMP_INST_SIZE;
    DLOG() << "Try to steal bytes: " << try_stolen_bytes;

    if (addr_distance < INT32_MAX && addr_distance > INT32_MIN) {
        // for 32 bit relative jump
        enable_mem_write(hook_mem);
        int64_t call_back_size =
            build_callback(hooked_func, hook_mem, 5, endbr_hooked);
        *trampoline_ptr = hook_mem;

        WLOG() << "the distance between hooked func and payload func: "
               << std::hex << addr_distance;
        const int32_t relative_addr_i32 = (int32_t)addr_distance;
        uint8_t *jmp_store_addr = endbr_hooked ? ((uint8_t *)hooked_func) + 4
                                               : (uint8_t *)hooked_func;
        memcpy(jmp_instrs + 1, &relative_addr_i32, sizeof(uint32_t));
        // memcpy(((uint8_t *)hooked_func) + 4, jmp_instrs, sizeof(jmp_instrs));
        memcpy(jmp_store_addr, jmp_instrs, sizeof(jmp_instrs));
    } else {
        // hook func is too far away, need to create a trampoline
        enable_mem_write(hook_mem);
        int64_t trampoline_size = build_callback(
            hooked_func, hook_mem, try_stolen_bytes, endbr_hooked);
        *trampoline_ptr = hook_mem;

        // create relay function
        void *relay_func_mem = (u_int8_t *)hook_mem + trampoline_size;
        write_absolute_jump64(relay_func_mem,
                              (uint8_t *)payload_func + endbr_hooked_bytes);
        DLOG() << "relay function address: " << std::hex << relay_func_mem;

        const int64_t relative_addr =
            (uint8_t *)relay_func_mem -
            ((uint8_t *)hooked_func + endbr_hooked_bytes + sizeof(jmp_instrs));
        DLOG() << "relative address from hooked function to relay function: "
               << std::hex << relative_addr;

        if (relative_addr > INT32_MAX || relative_addr < INT32_MIN) {
            WLOG() << "relative address is too large, use absolute jump from "
                      "hooked function to relay function";
            write_absolute_jump64(hooked_func, relay_func_mem);
        } else {
            WLOG() << "relative address is in 32 bit range, use relative jump "
                      "from hooked function to relay function";
            const int32_t relative_addr_u32 = (int32_t)relative_addr;
            memcpy(jmp_instrs + 1, &relative_addr_u32, sizeof(uint32_t));
            memcpy(((uint8_t *)hooked_func) + endbr_hooked_bytes, jmp_instrs,
                   sizeof(jmp_instrs));
        }
    }
}

void LocalHookRegistrar::add(LocalHookInfo info) { hooks.emplace_back(info); }

void LocalHookRegistrar::install() {
    auto lib_vec = cfunc_hook::get_libs();
    for (auto &lib_name : lib_vec) {
        DLOG() << "lib name: " << lib_name;
        void *handle = dlopen(lib_name.c_str(), RTLD_LAZY);
        int64_t hook_size = hooks.size();
        for (int64_t index = 0; index < hook_size; index++) {
            if (hooks[index].installed) {
                continue;
            }
            void *func_ptr = dlsym(handle, hooks[index].symbol.c_str());
            if (func_ptr) {
                LOG() << "hook function symbol: " << hooks[index].symbol;
                install_local_hook(func_ptr, hooks[index].new_func,
                                   hooks[index].trampoline);
                hooks[index].installed = true;
            }
        }
    }
}

typedef utils::Singleton<LocalHookRegistrar> LocalHookRegistrarSingleton;

LocalHookRegistration::LocalHookRegistration(std::string symbol, void *new_func,
                                             void **trampoline) {
    LocalHookRegistrarSingleton::instance().get_elem()->add(
        {symbol, new_func, trampoline, false});
}

void install_local_hooks() {
    LocalHookRegistrarSingleton::instance().get_elem()->install();
}

} // namespace local_hook