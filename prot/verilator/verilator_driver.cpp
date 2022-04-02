// initially will be similar to src/comp_driver.sv, but we will extend this.

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <verilated.h>
#include <vector>
#include <verilated_vcd_c.h>
#include "comp.h"

#define MAX_SIM_TIME 1000
vluint64_t sim_time = 0;

double sc_time_stamp() {
    return sim_time;
}

int main(int argc, char **argv, char **env)
{
    comp *dut = new comp;

    // Verilated::traceEverOn(true);
    // VerilatedVcdC *m_trace = new VerilatedVcdC;
    // dut->trace(m_trace, 5);
    // m_trace->open("waveform.vcd");

    dut->rst = 1;
    dut->oob_wen = 0;
    dut->ena = 0;
    dut->clk = 0;
    sim_time++;
    dut->eval();
    dut->clk = 1;
    sim_time++;
    dut->eval();
    dut->rst = 0;
    dut->clk = 0;
    sim_time++;
    dut->eval();
    dut->clk = 1;
    sim_time++;
    dut->eval();

    std::fstream infile;
    infile.open("../../../build/prog.hex", std::fstream::in);
    unsigned int a;
    unsigned int addr = 0;
    while(infile >> std::hex >> a) {
        // std::cout << a << std::endl;
        dut->oob_wen = 1;
        dut->oob_wr_addr = addr;
        dut->oob_wr_data = a;
        addr += 4;
        dut->clk = 0;
        sim_time++;
        dut->eval();
        dut->clk = 1;
        sim_time++;
        dut->eval();
    }

    dut->oob_wen = 0;
    dut->ena = 1;
    dut->clk = 0;
    sim_time++;
    dut->eval();
    dut->eval();
    dut->eval();
    dut->clk = 1;
    sim_time++;
    dut->eval();
    dut->eval();
    dut->eval();

    std::vector<unsigned int> outs;

    while (sim_time < MAX_SIM_TIME)
    {
        dut->clk = 0;
        dut->eval();
        dut->eval();
        dut->eval();
        dut->clk = 1;
        sim_time++;
        dut->eval();
        dut->eval();
        dut->eval();
        // std::cout << int(dut->halt) << int(dut->outen) <<  std::endl;
        if(int(dut->halt)) {
            std::cout << "HALT" << std::endl;
            break;
        }
        if(int(dut->outen)) {
            // std::cout << "OUT " << int(dut->out) << std::endl;
            outs.push_back(int(dut->out));
        }
        // m_trace->dump(sim_time);
        sim_time++;
    }

    unsigned i = 0;
    for(auto it = begin(outs); it != end(outs); ++it) {
        std::cout << "out " << i << " " << *it << std::endl;
        i += 1;
    }

    // m_trace->close();
    delete dut;
    exit(EXIT_SUCCESS);
}
