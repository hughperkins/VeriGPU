cmake_minimum_required(VERSION 3.22)

project(prot_init)
find_package(verilator HINTS $ENV{VERILATOR_ROOT})
add_executable(prot_init ../prot_init.cpp ${VERILATOR_ROOT}/include/verilated.cpp)
verilate(prot_init SOURCES prot_init.sv
    TOP_MODULE prot_init
    PREFIX prot_init
    VERILATOR_ARGS +1800-2017ext+sv --report-unoptflat
)
