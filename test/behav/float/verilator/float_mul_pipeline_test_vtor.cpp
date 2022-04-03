#include <iostream>
#include "float_mul_pipeline_test_vtor.h"

static float_mul_pipeline_test_vtor *dut;

void dut_new() {
    dut = new float_mul_pipeline_test_vtor;
}
void dut_delete() {
    delete dut;
}
unsigned char &dut_clk() {
    return dut->clk;
}
unsigned char &dut_ack() {
    return dut->ack;
}
unsigned char &dut_rst() {
    return dut->rst;
}
unsigned char &dut_finish() {
    return dut->finish;
}
unsigned char &dut_fail() {
    return dut->fail;
}
unsigned int &dut_cnt() {
    return dut->cnt;
}
unsigned int &dut_out() {
    return dut->out;
}
unsigned int &dut_test_num() {
    return dut->test_num;
}
void dut_eval() {
    dut->eval();
}
