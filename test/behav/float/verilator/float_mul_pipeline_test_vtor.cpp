// initially will be similar to src/comp_driver.sv, but we will extend this.

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <verilated.h>
#include <vector>
#include <bitset>
#include <iomanip>
#include <verilated_vcd_c.h>
#include "float_mul_pipeline_test_vtor.h"

#define MAX_SIM_TIME 10000
// #define MAX_SIM_TIME 2000
vluint64_t sim_time = 0;

double sc_time_stamp() {
    return sim_time;
}

int main(int argc, char **argv, char **env)
{
    Verilated::commandArgs(argc, argv);

    float_mul_pipeline_test_vtor *dut = new float_mul_pipeline_test_vtor;

    Verilated::assertOn(false);

    dut->rst = 1;
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();

    dut->rst = 0;
    dut->clk = 0;
    dut->eval();
    Verilated::assertOn(true);
    dut->clk = 1;
    dut->eval();

    while(sim_time < MAX_SIM_TIME && !int(dut->finish)) {
        dut->clk = 0;
        sim_time += 5;
        dut->eval();
        dut->clk = 1;
        sim_time += 5;
        dut->eval();
        unsigned int out = (unsigned int)(dut->out);
        if(int(dut->ack)) {
        std::cout << "t=" << sim_time << " test_num=" << int(dut->test_num) << " cnt=" << int(dut->cnt) <<
            " ack=" << int(dut->ack) << " out=" << (reinterpret_cast<float &>(out)) << " fail=" << int(dut->fail) <<
            " finish=" << int(dut->finish) << std::endl;
        }
    }

    assert(int(dut->finish));
    assert(~int(dut->fail));
    std::cout << "finished" << std::endl;

    delete dut;
    exit(EXIT_SUCCESS);
}
