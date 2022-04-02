// initially will be similar to src/comp_driver.sv, but we will extend this.

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <verilated.h>
#include <vector>
#include <bitset>
#include <verilated_vcd_c.h>
#include "comp.h"

#define MAX_SIM_TIME 5000000
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
    sim_time += 5;
    dut->eval();
    dut->clk = 1;
    sim_time += 5;
    dut->eval();
    dut->rst = 0;
    dut->clk = 0;
    sim_time += 5;
    dut->eval();
    dut->clk = 1;
    sim_time += 5;
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
        sim_time += 5;
        dut->eval();
        dut->clk = 1;
        sim_time += 5;
        dut->eval();
    }

    dut->oob_wen = 0;
    dut->ena = 1;
    dut->clk = 0;
    sim_time += 5;
    // dut->eval();
    // dut->eval();
    dut->eval();
    dut->clk = 1;
    sim_time += 5;
    // dut->eval();
    // dut->eval();
    dut->eval();

    std::vector<unsigned int> outs;
    std::vector<int> out_types;
    std::vector<float> out_floats;

    while (sim_time < MAX_SIM_TIME)
    {
        dut->clk = 0;
        sim_time += 5;
        dut->eval();
        // dut->eval();
        // dut->eval();
        dut->clk = 1;
        sim_time += 5;
        dut->eval();
        // dut->eval();
        // dut->eval();
        // std::cout << int(dut->halt) << int(dut->outen) <<  std::endl;
        if(int(dut->halt)) {
            std::cout << "HALT" << std::endl;
            break;
        }
        if(int(dut->outen)) {
            out_types.push_back(0);
            std::cout << "OUT " << int(dut->out) << std::endl;
            outs.push_back(int(dut->out));
        }
        if(int(dut->outflen)) {
            out_types.push_back(1);
            std::cout << "OUT.S " << reinterpret_cast<float &>(dut->out) << std::endl;
            out_floats.push_back(reinterpret_cast<float &>(dut->out));
            std::bitset<32> bits(dut->out);
            std::cout << "OUT.S" << bits << std::endl;
            // out_floats.push_back((float)(dut->out));
        }
        // m_trace->dump(sim_time);
        // sim_time++;
    }

    unsigned i = 0;
    int int_i = 0;
    int float_i = 0;
    for(auto it = begin(out_types); it != end(out_types); ++it) {
        // std::cout << i << " " << *it << std::endl;
        if(*it) {
            std::bitset<32> bits(reinterpret_cast<int &>(out_floats[float_i]));
            std::cout << "out.s " << i << " " << bits << " " << std::fixed << std::setprecision(6) << out_floats[float_i] << std::endl;
            float_i++;
        } else {
            std::bitset<32> bits(outs[int_i]);
            std::cout << "out " << i << " " << bits << " " << std::hex << std::setw(8) << std::setfill('0') << outs[int_i] << " " << std::dec << outs[int_i] << std::endl;
            int_i++;
        }
        i += 1;
    }
    // for(auto it = begin(outs); it != end(outs); ++it) {
    //     std::bitset<32> bits(*it);
    //     std::cout << "out " << i << " " << bits << " " << std::hex << std::setw(8) << std::setfill('0') << *it << " " << std::dec << *it << std::endl;
    //     i += 1;
    // }

    // for(auto it = begin(out_floats); it != end(out_floats); ++it) {
    //     std::cout << "out.s " << i << *it << std::endl;
    //     // std::bitset<32> bits(*it);
    //     // std::cout << "out " << i << " " << bits << " " << std::hex << std::setw(8) << std::setfill('0') << *it << " " << std::dec << *it << std::endl;
    //     i += 1;
    // }

    // m_trace->close();
    delete dut;
    exit(EXIT_SUCCESS);
}
