// initially will be similar to src/comp_driver.sv, but we will extend this.

#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <verilated.h>
#include <vector>
#include <bitset>
#include <verilated_vcd_c.h>
#include "prot_unified_source.h"

#define MAX_SIM_TIME 5000000
vluint64_t sim_time = 0;

double sc_time_stamp() {
    return sim_time;
}

uint totalMemoryBytes = 1024;

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

void *gpuMalloc(unsigned int requestedBytes) {
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

void gpuCopy(void *gpuMemPtr, void *srcData, size_t numBytes) {

}

int main(int argc, char **argv, char **env)
{
    prot_unified_source *dut = new prot_unified_source;

    MemoryInfo *p_memInfo = new MemoryInfo(0, totalMemoryBytes);
    freeSpaces.insert(p_memInfo);

    dut->rst = 0;
    dut->oob_wen = 0;
    dut->ena = 0;
    dut->clk = 0;
    sim_time += 5;
    dut->eval();
    dut->clk = 1;
    sim_time += 5;
    dut->eval();

    dut->rst = 1;
    // dut->clk = 0;
    // sim_time += 5;
    // dut->eval();
    // dut->clk = 1;
    // sim_time += 5;
    // dut->eval();

    // first copy data in
    // we'll just make up our own commands for now
    unsigned int values[] = {3,1,7,9,11};
    unsigned int numValues = 5;
    void *ptrMemory = gpuMalloc(numValues * sizeof(unsigned int));
    std::cout << "found memory at " << ptrMemory << std::endl;

    gpuCopy(ptrMemory, values, numValues * sizeof(unsigned int));
    // copyToGpu(values, );
    return 0;

    delete prot_unified_source;
    exit(EXIT_SUCCESS);
}
