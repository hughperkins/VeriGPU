#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <set>
#include <verilated.h>
#include <vector>
#include <bitset>
#include <verilated_vcd_c.h>
#include "gpu_card.h"

#include "gpu_runtime.h"

#define MAX_SIM_TIME 5000000
// #define MAX_SIM_TIME 250
vluint64_t sim_time = 0;

double sc_time_stamp()
{
    return sim_time;
}

uint32_t totalMemoryBytes = 1024; // yes we need to move to 64-bits soonish...

static gpu_card *dut = 0;

// std::vector<MemoryInfo *> freeSpaces;
static std::set<MemoryInfo *> freeSpaces;
static std::set<MemoryInfo *> usedSpaces;

enum e_instr
{
    NOP = 0,
    COPY_TO_GPU = 1,
    COPY_FROM_GPU = 2,
    KERNEL_LAUNCH = 3
};

void tick()
{
    dut->clk = 0;
    sim_time += 5;
    dut->eval();
    dut->clk = 1;
    sim_time += 5;
    dut->eval();
}

void gpuCreateContext()
{
    dut = new gpu_card;
    dut->rst = 0;
    // dut->oob_wen = 0;
    // dut->ena = 0;
    dut->cpu_recv_instr = NOP;

    tick();

    dut->rst = 1;
    tick();

    // we skip the first 32 words, so that we can have void pointers
    MemoryInfo *p_memInfo = new MemoryInfo(128, totalMemoryBytes - 128);
    freeSpaces.insert(p_memInfo);
}

void gpuDestroyContext()
{
    std::cout << "gpuDestroyContext before delete dut" << std::endl;
    delete dut;
    std::cout << "gpuDestroyContext after delete dut" << std::endl;
    dut = 0;
}

std::ostream &operator<<(std::ostream &os, const MemoryInfo &mi)
{
    os << "MemoryInfo(pos=" << mi.pos << ", size=" << mi.size << ")";
    return os;
}

void *gpuMalloc(uint32_t requestedBytes)
{
    // who should manage the memory? driver? gpu?
    // maybe driver???
    std::cout << "gpuMalloc " << requestedBytes << std::endl;
    MemoryInfo *freeSpace = 0;
    for (auto it = freeSpaces.begin(), e = freeSpaces.end(); it != e; it++)
    {
        MemoryInfo *candidate = *it;
        if (candidate->size >= requestedBytes)
        {
            freeSpace = candidate;
            break;
        }
    }
    std::cout << "freeSpace " << freeSpace << std::endl;
    if (freeSpace == 0)
    {
        throw std::runtime_error(std::string("Not enough free space"));
    }
    std::cout << "found freeSpace" << *freeSpace << std::endl;
    if (freeSpace->size > requestedBytes + 256)
    {
        std::cout << "splitting, since available chunk is size " << freeSpace->size
                  << " and we only need " << requestedBytes << std::endl;
        MemoryInfo *secondSpace = new MemoryInfo(freeSpace->pos + requestedBytes, freeSpace->size - requestedBytes);
        std::cout << "new spaces " << *freeSpace << " " << *secondSpace << std::endl;
        freeSpaces.erase(freeSpace);
        freeSpaces.insert(secondSpace);
        usedSpaces.insert(freeSpace);
        return (void *)freeSpace->pos;
    }
    else
    {
        freeSpaces.erase(freeSpace);
        usedSpaces.insert(freeSpace);
        return (void *)freeSpace->pos;
    }
}

void gpuCopyToDevice(void *gpuMemPtr, void *srcData, size_t numBytes)
{
    std::cout << "gpuCopyToDevice our addr " << srcData << " theirs " << gpuMemPtr << " numBytes " << numBytes << std::endl;
    dut->cpu_recv_instr = COPY_TO_GPU;
    tick();

    dut->cpu_in_data = (uint32_t)(size_t)gpuMemPtr;
    tick();

    dut->cpu_in_data = (uint32_t)numBytes;
    tick();

    dut->cpu_recv_instr = NOP;
    uint32_t *srcDataWords = (uint32_t *)srcData;
    long numWords = numBytes >> 2;
    for (long i = 0; i < numWords; i++)
    {
        std::cout << "sending word " << i << " which is " << srcDataWords[i] << std::endl;
        dut->cpu_in_data = srcDataWords[i];
        tick();
    }
    std::cout << "hopefully copied data to GPU" << std::endl;
}

void gpuCopyFromDevice(void *destData, void *gpuMemPtr, size_t numBytes)
{
    std::cout << "gpuCopyFromDevice our addr " << destData << " theirs " << gpuMemPtr << " numBytes " << numBytes << std::endl;
    dut->cpu_recv_instr = COPY_FROM_GPU;
    tick();

    dut->cpu_in_data = (uint32_t)(size_t)gpuMemPtr;
    tick();

    dut->cpu_in_data = (uint32_t)numBytes;
    tick();

    dut->cpu_recv_instr = NOP;
    uint32_t *destDataWords = (uint32_t *)destData;
    long numWords = numBytes >> 2;
    std::cout << "gpuCopyFromDevice numWords=" << numWords << " sim_time=" << sim_time << std::endl;
    long i = 0;
    while(i < numWords && sim_time < MAX_SIM_TIME) {
        std::cout << "gpuCopyFromDevice i=" << i << " sim_time=" << sim_time << std::endl;
        if (dut->cpu_out_ack) {
            destDataWords[i] = dut->cpu_out_data;
            std::cout << "gpuCopyFromDevice received word " << i << " which is " << destDataWords[i] << std::endl;
            i++;
        }
        tick();
    }
    // for (long i = 0; i < numWords; i++)
    // {
    //     tick();
    //     destDataWords[i] = dut->cpu_out_data;
    //     std::cout << "received word " << i << " which is " << destDataWords[i] << std::endl;
    // }
    std::cout << "hopefully received data from GPU" << std::endl;
}

void gpuLaunchKernel(void *kernelPos, uint32_t numParams, const uint32_t *const p_params) {
    dut->cpu_recv_instr = KERNEL_LAUNCH;
    tick();

    dut->cpu_in_data = (size_t)kernelPos;
    tick();

    dut->cpu_in_data = numParams;
    tick();

    for(int i = 0; i < numParams; i++) {
        dut->cpu_in_data = p_params[i];
        tick();
    }
    dut->cpu_recv_instr = NOP;

    while (!dut->cpu_out_ack)
    {
        std::cout << "gpu_runtime.cpp awaiting cpu_out_ack" << std::endl;
        tick();
    }
    std::cout << "gpu_runtime.cpp gpuLaunchKernel kernel finished" << std::endl;
}
