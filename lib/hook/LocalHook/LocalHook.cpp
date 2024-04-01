#include "hook/LocalHook/LocalHook.h"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

#include "capstone/capstone.h"
#include "capstone/x86.h"

// #include "../Log/LogStream.h"

namespace local_hook {

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

void *alloc_page_near_address(void *target_addr) {
    static uint64_t page_size = getpagesize();
    uintptr_t aligned_addr = ((uintptr_t)target_addr) & (~(page_size - 1));
    uintptr_t free_mem = find_free_address((uintptr_t)aligned_addr, page_size);
    void *mmap_addr = mmap((void *)free_mem, page_size, PROT_READ | PROT_EXEC,
                           MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mmap_addr == MAP_FAILED) {
        perror("mmap");
        std::cout << "mmap failed" << std::endl;
        exit(0);
    }
    std::cout << "mmap success: " << mmap_addr << std::endl;
    return mmap_addr;
}

struct X64Instructions {
    cs_insn *instructions;
    uint32_t numInstructions;
    uint32_t numBytes;
};

int64_t check_func_mem(void *function) {
    csh handle;
    auto s = cs_open(CS_ARCH_X86, CS_MODE_64, &handle);
    if (s != 0) {
        std::cout << "Error opening capstone handle" << std::endl;
    }
    s = cs_option(handle, CS_OPT_DETAIL,
                  CS_OPT_ON); // we need details enabled for relocating RIP
                              // relative instrs
    if (s != 0) {
        std::cout << "Error set option" << std::endl;
    }

    uint32_t byte_count = 0;

    // for cpu with BIT check, the first instruction of a function is endbr
    // endbr64 instruction: 0xfa1e0ff3
    uint32_t endbr64 = 0xfa1e0ff3;
    uint8_t *code = (uint8_t *)function;
    if (endbr64 == *(uint32_t *)function) {
        code = (uint8_t *)function + 4;
        byte_count += 4;
    }

    cs_insn *disassembled_instrs = nullptr;
    size_t count =
        cs_disasm(handle, code, 50, (uint64_t)code, 0, &disassembled_instrs);
    if (count == 0) {
        s = cs_errno(handle);
        std::cout << "error status: " << cs_strerror(s) << std::endl;
    }

    for (int32_t i = 0; i < count; ++i) {
        cs_insn &inst = disassembled_instrs[i];
        byte_count += inst.size;
    }

    cs_free(disassembled_instrs, count);
    cs_close(&handle);
    return byte_count;
}

X64Instructions steal_bytes(void *function, int64_t bytes) {
    // Disassemble stolen bytes
    std::cout << "Stealing bytes from: " << function << std::endl;
    csh handle;
    auto s = cs_open(CS_ARCH_X86, CS_MODE_64, &handle);
    if (s != 0) {
        std::cout << "Error opening capstone handle" << std::endl;
    }
    s = cs_option(handle, CS_OPT_DETAIL,
                  CS_OPT_ON); // we need details enabled for relocating RIP
                              // relative instrs
    if (s != 0) {
        std::cout << "Error set option" << std::endl;
    }

    // for cpu with BIT check, the first instruction of a function is endbr
    // endbr64 instruction: 0xfa1e0ff3
    uint32_t endbr64 = 0xfa1e0ff3;
    uint8_t *code = (uint8_t *)function;
    if (endbr64 == *(uint32_t *)function) {
        code = (uint8_t *)function + 4;
    }

    cs_insn *disassembled_instrs = nullptr;
    size_t count = cs_disasm(handle, code, 20 + bytes, (uint64_t)code,
                             20 + bytes, &disassembled_instrs);
    if (count == 0) {
        s = cs_errno(handle);
        std::cout << "error status: " << cs_strerror(s) << std::endl;
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

    // std::cout << std::hex << (void *)code << std::endl;
    // replace instructions in target func wtih NOPs
    memset((void *)code, 0x90, byte_count);

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
    memcpy(&abs_call_instrs[6], &call_target_addr, sizeof(call_target_addr));
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
    switch (bytes) {
    case 1: {
        return INT8_MIN < offset < INT8_MAX;
    }
    case 2: {
        return INT16_MIN < offset < INT16_MAX;
    }
    case 4: {
        return INT32_MIN < offset < INT32_MAX;
    }
    default: {
        std::cout << "Unsupported operand size: " << bytes << std::endl;
        exit(0);
        return false;
    }
    }
    return false;
}

/// @brief instruction has been move to a new address,
/// so need to update the operand which is relative address.
/// @param inst
/// @param new_addr
void relocate_instruction(cs_insn *inst, void *new_addr) {
    cs_x86 *x86 = &(inst->detail->x86);
    uint64_t inst_addr = inst->address;
    // 1. traverse the operands of instruction, find the one that is relative
    // address
    for (uint32_t i = 0; i < x86->op_count; ++i) {
        cs_x86_op *op = &(x86->operands[i]);
        if (op->type == X86_OP_MEM && op->mem.base == X86_REG_RIP) {
            uint8_t bytes = op->size;
            int64_t disp = op->mem.disp;
            disp -= (int64_t)((uint64_t)new_addr - inst_addr);
            if (!check_mem_offset(disp, bytes)) {
                std::cout << "Invalid displacement: " << disp << std::endl;
                exit(0);
            }
            op->mem.disp = disp;
        }
    }
}

bool is_jump(cs_insn &inst) {
    if (inst.id >= X86_INS_JAE && inst.id <= X86_INS_JS) {
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

uint32_t add_jmp_to_abs_table(cs_insn &inst, uint8_t *write_addr) {
    // char* jmp_target_addr = (char*)inst.op_str;
    if (!is_jump(inst)) {
        std::cout << " not a jump instruction: " << inst.mnemonic << std::endl;
        exit(0);
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
        std::cout << "Invalid jmp offset: " << jmp_offset << std::endl;
        exit(0);
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

uint32_t add_call_to_abs_table(cs_insn &inst, uint8_t *write_addr,
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
        std::cout << "Invalid jmp offset: " << jmp_offset << std::endl;
        exit(0);
    }
    uint8_t u8_jmp_offset = jmp_offset;
    // construct jmp instruction
    uint8_t jmp_bytes[2] = {0xEB, u8_jmp_offset};
    uint8_t nop = 0x90;
    memset(inst.bytes, nop, inst.size);
    memcpy(inst.bytes, jmp_bytes, sizeof(jmp_bytes));
}

/**
 *       __________________________________
 *      |______________endbr64_____________|
 *      |______________......._____________|------|
 *      |______________......._____________|      |--> stolen instructions
 *      |______________......._____________|------|
 *      |______________endbr64_____________| 4 bytes
 *      |__________mov r10, addr___________| 10 bytes = 2bytes + 8 bytes
 *      |__________jmpq r10________________| 3 bytes
 *      |__________________________________|
 **/
int64_t build_callback(void *func2hook, void *hook_mem, int64_t bytes) {
    X64Instructions stolenInstrs = steal_bytes(func2hook, bytes);

    uint64_t endbr64_bytes = 4;
    uint64_t mov_bytes = 10;
    uint64_t jmp_abs_bytes = 3;
    uint8_t *stolenByteMem = (uint8_t *)hook_mem;
    uint8_t *jumpBackMem =
        stolenByteMem + endbr64_bytes + stolenInstrs.numBytes;
    std::cout << "stolen bytes: " << stolenInstrs.numBytes << std::endl;
    uint8_t *absTableMem = jumpBackMem + mov_bytes + jmp_abs_bytes;

    uint8_t endbr_instr[4] = {0xf3, 0x0f, 0x1e, 0xfa};
    memcpy(hook_mem, endbr_instr, sizeof(endbr_instr));
    stolenByteMem += sizeof(endbr_instr);
    std::cout << "stolen instructions number: " << stolenInstrs.numInstructions
              << std::endl;

    for (uint32_t i = 0; i < stolenInstrs.numInstructions; ++i) {
        cs_insn &inst = stolenInstrs.instructions[i];
        if (inst.id >= X86_INS_LOOP && inst.id <= X86_INS_LOOPNE) {
            return 0; // bail out on loop instructions, I don't have a good way
                      // of handling them
        }

        if (is_rip_relative_inst(inst)) {
            std::cout << "rip relative instruction: " << std::endl;
            relocate_instruction(&inst, stolenByteMem);
        } else if (is_relative_jump(inst)) {
            uint64_t abs_jmp_size = add_jmp_to_abs_table(inst, absTableMem);
            rewrite_jmp_instruction(inst, stolenByteMem, absTableMem);
            absTableMem += abs_jmp_size;
        } else if (inst.id == X86_INS_CALL) {
            // uint32_t abs_call_size = add_call_to_abs_table(inst, absTableMem,
            // jumpBackMem);
            uint8_t *jump_back_addr = stolenByteMem + inst.size;
            uint32_t abs_call_size =
                add_call_to_abs_table(inst, absTableMem, jump_back_addr);
            rewrite_call_instruction(inst, stolenByteMem, absTableMem);
            absTableMem += abs_call_size;
        }
        std::cout << "memory to store stolen bytes: " << std::hex
                  << (void *)stolenByteMem << std::endl;
        memcpy(stolenByteMem, inst.bytes, inst.size);
        stolenByteMem += inst.size;
    }

    write_absolute_jump64(jumpBackMem, (uint8_t *)func2hook + endbr64_bytes +
                                           stolenInstrs.numBytes);
    free(stolenInstrs.instructions);
    return absTableMem - (uint8_t *)hook_mem;
}

void install_local_hook(void *hooked_func, void *payload_func,
                  void **trampoline_ptr) {
    uint8_t jmp_instrs[5] = {0xE9, 0x0, 0x0, 0x0, 0x0};
    enable_mem_write(hooked_func);
    int64_t addr_distance =
        ((uint64_t)payload_func + 4) - ((uint64_t)hooked_func + 9);
    if (INT32_MIN < addr_distance < INT32_MAX) {
        // for 32 bit relative jump
        void *hook_mem = alloc_page_near_address(hooked_func);
        enable_mem_write(hook_mem);
        int64_t call_back_size = build_callback(hooked_func, hook_mem, 5);
        std::cout << "hookd_mem: " << std::hex << hook_mem << std::endl;
        *trampoline_ptr = hook_mem;

        std::cout << "hook distance is in 32 bit range" << std::endl;
        const int32_t relative_addr_i32 = addr_distance;
        std::cout << "relative addr: " << std::hex << relative_addr_i32
                  << std::endl;
        memcpy(jmp_instrs + 1, &relative_addr_i32, sizeof(uint32_t));
        memcpy(((uint8_t *)hooked_func) + 4, jmp_instrs, sizeof(jmp_instrs));
    } else {
        // hook func is too far away, need to create a trampoline
        void *hook_mem = alloc_page_near_address(hooked_func);
        enable_mem_write(hook_mem);
        int64_t trampoline_size = build_callback(hooked_func, hook_mem, 5);
        std::cout << "callback size: " << std::dec << trampoline_size
                  << std::endl;
        *trampoline_ptr = (uint8_t *)hook_mem;

        // create relay function
        void *relay_func_mem = (u_int8_t *)hook_mem + trampoline_size;
        write_absolute_jump64(relay_func_mem, (uint8_t *)payload_func + 4);

        const uint64_t relative_addr =
            (uint8_t *)relay_func_mem -
            ((uint8_t *)hooked_func + 4 + sizeof(jmp_instrs));
        if (relative_addr > UINT32_MAX) {
            std::cout << "Error: relative address is larger than UINT32_MAX"
                      << std::endl;
        }
        const uint32_t relative_addr_u32 = (uint32_t)relative_addr;
        memcpy(jmp_instrs + 1, &relative_addr_u32, sizeof(uint32_t));
        memcpy(((uint8_t *)hooked_func) + 4, jmp_instrs, sizeof(jmp_instrs));
    }
}

} // namespace local_hook