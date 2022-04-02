// initially will be similar to src/comp_driver.sv, but we will extend this.

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <verilated.h>
#include <vector>
#include <bitset>
#include <verilated_vcd_c.h>

#include "prot_init.h"

#define MAX_SIM_TIME 5000000
vluint64_t sim_time = 0;

double sc_time_stamp() {
    return sim_time;
}

int main(int argc, char **argv, char **env)
{
    Verilated::commandArgs(argc, argv);

    prot_init *dut = new prot_init;

    std::cout << int(dut->out) << std::endl;
    dut->eval();
    std::cout << int(dut->out) << std::endl;
    dut->eval();
    std::cout << int(dut->out) << std::endl;

    delete dut;
    exit(EXIT_SUCCESS);
}
