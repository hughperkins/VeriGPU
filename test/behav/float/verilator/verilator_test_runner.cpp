// initially will be similar to src/comp_driver.sv, but we will extend this.

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <verilated.h>
#include <vector>
#include <bitset>
#include <iomanip>
#include <verilated_vcd_c.h>
// #include "float_mul_pipeline_test_vtor.h"

#define MAX_SIM_TIME 10000
// #define MAX_SIM_TIME 2000
vluint64_t sim_time = 0;

double sc_time_stamp() {
    return sim_time;
}

unsigned char &dut_clk();
unsigned char &dut_ack();
unsigned char &dut_rst();
unsigned char &dut_fail();
unsigned char &dut_finish();
unsigned int &dut_cnt();
unsigned int &dut_out();
unsigned int &dut_test_num();
void dut_eval();
void dut_new();
void dut_delete();

int main(int argc, char **argv, char **env)
{
    Verilated::commandArgs(argc, argv);

    dut_new();

    Verilated::assertOn(false);

    dut_rst() = 0;
    dut_clk() = 0;
    dut_eval();
    dut_clk() = 1;
    dut_eval();

    dut_rst() = 1;
    dut_clk() = 0;
    dut_eval();
    Verilated::assertOn(true);
    dut_clk() = 1;
    dut_eval();

    while(sim_time < MAX_SIM_TIME && !int(dut_finish())) {
        dut_clk() = 0;
        sim_time += 5;
        dut_eval();
        dut_clk() = 1;
        sim_time += 5;
        dut_eval();
        unsigned int out = (unsigned int)(dut_out());
        if(int(dut_ack())) {
        std::cout << "t=" << sim_time << " test_num=" << int(dut_test_num()) << " cnt=" << int(dut_cnt()) <<
            " ack=" << int(dut_ack()) << " out=" << (reinterpret_cast<float &>(out)) << " fail=" << int(dut_fail()) <<
            " finish=" << int(dut_finish()) << std::endl;
        }
    }

    assert(int(dut_finish()));
    assert(~int(dut_fail()));
    std::cout << "finished" << std::endl;

    dut_delete();
    exit(EXIT_SUCCESS);
}
