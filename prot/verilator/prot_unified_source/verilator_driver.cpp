// initially will be similar to src/comp_driver.sv, but we will extend this.

#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <verilated.h>
#include <vector>
#include <bitset>
#include <verilated_vcd_c.h>
#include "controller.h"

#define MAX_SIM_TIME 5000000
vluint64_t sim_time = 0;

double sc_time_stamp() {
    return sim_time;
}

uint32_t totalMemoryBytes = 1024;  // yes we need to move to 64-bits soonish...

enum e_instr {
    NOP = 0,
    COPY_TO_GPU = 1,
    COPY_FROM_GPU = 2,
    KERNEL_LAUNCH = 3
};

void tick(controller *dut) {
    dut->clk = 0;
    sim_time += 5;
    dut->eval();
    dut->clk = 1;
    sim_time += 5;
    dut->eval();
}

class MemoryInfo {
public:
    MemoryInfo(size_t pos, size_t size) : pos(pos), size(size) {}
    ~MemoryInfo() {}
    friend std::ostream& operator<<(std::ostream& os, const MemoryInfo& dt);

// protected:
    size_t pos;
    size_t size;
};

std::ostream& operator<<(std::ostream& os, const MemoryInfo& mi)
{
    os << "MemoryInfo(pos=" << mi.pos << ", size=" << mi.size << ")";
    return os;
}

// std::vector<MemoryInfo *> freeSpaces;
std::set<MemoryInfo *> freeSpaces;
std::set<MemoryInfo *> usedSpaces;

void *gpuMalloc(uint32_t requestedBytes) {
    // who should manage the memory? driver? gpu?
    // maybe driver???
    MemoryInfo *freeSpace = 0;
    for(auto it=freeSpaces.begin(), e=freeSpaces.end(); it != e; it++) {
        MemoryInfo *candidate = *it;
        if(candidate->size >= requestedBytes) {
            freeSpace = candidate;
            break;
        }
    }
    if(freeSpace == 0) {
        throw std::runtime_error(std::string("Not enough free space"));
    }
    std::cout << "found freeSpace" << *freeSpace << std::endl;
    if(freeSpace->size > requestedBytes + 256) {
        std::cout << "splitting, since available chunk is size " << freeSpace->size
            << " and we only need " << requestedBytes << std::endl;
        MemoryInfo *secondSpace = new MemoryInfo(freeSpace->pos + requestedBytes, freeSpace->size - requestedBytes);
        std::cout << "new spaces " << *freeSpace << " " << *secondSpace << std::endl;
        freeSpaces.erase(freeSpace);
        freeSpaces.insert(secondSpace);
        usedSpaces.insert(freeSpace);
        return (void *)freeSpace->pos;
    } else {
        freeSpaces.erase(freeSpace);
        usedSpaces.insert(freeSpace);
        return (void *)freeSpace->pos;
    }
}

void gpuCopy(controller *dut, void *gpuMemPtr, void *srcData, size_t numBytes) {
    std::cout << "gpuCopy our addr " << srcData << " theirs " << gpuMemPtr << " numBytes " << numBytes << std::endl;
    dut->recv_instr = COPY_TO_GPU;
    tick(dut);

    dut->in_data = (uint32_t)(size_t)gpuMemPtr;
    tick(dut);

    dut->in_data = (uint32_t)numBytes;
    tick(dut);

    dut->recv_instr = NOP;
    uint32_t *srcDataWords = (uint32_t *)srcData;
    long numWords = numBytes >> 2;
    for(long i = 0; i < numWords; i++) {
        std::cout << "sending word " << i << " which is " << srcDataWords[i] << std::endl;
        dut->in_data = srcDataWords[i];
        tick(dut);
    }
    std::cout << "hopefully copied data to GPU" << std::endl;
}

int main(int argc, char **argv, char **env)
{
    controller *dut = new controller;

    MemoryInfo *p_memInfo = new MemoryInfo(0, totalMemoryBytes);
    freeSpaces.insert(p_memInfo);

    dut->rst = 0;
    // dut->oob_wen = 0;
    // dut->ena = 0;
    dut->recv_instr = NOP;

    tick(dut);

    dut->rst = 1;
    tick(dut);

    // dut->clk = 0;
    // sim_time += 5;
    // dut->eval();
    // dut->clk = 1;
    // sim_time += 5;
    // dut->eval();

    // first copy data in
    // we'll just make up our own commands for now
    uint32_t values[] = {3,1,7,9,11};
    uint32_t numValues = 5;
    void *ptrMemory = gpuMalloc(numValues * sizeof(uint32_t));
    std::cout << "found memory at " << ptrMemory << std::endl;

    gpuCopy(dut, ptrMemory, values, numValues * sizeof(uint32_t));
    // copyToGpu(values, );

    delete dut;
    exit(EXIT_SUCCESS);
}
