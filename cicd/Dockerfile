# this is used for cicd

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y gperf flex bison build-essential clang tcl-dev libboost-dev autoconf wget pkg-config \
    libfl-dev zlibc zlib1g zlib1g-dev perl git ccache libgoogle-perftools-dev numactl \
    perl-doc python3 python3-pip python-is-python3 libreadline-dev cmake
RUN python3 -V

# iverilog

# old
# RUN wget ftp://ftp.icarus.com/pub/eda/verilog//v11/verilog-11.0.tar.gz && tar -xzvf verilog-11.0.tar.gz &&     cd verilog-11.0 && ./configure && make -j $(nproc) && make install && cd .. && rm -Rf verilog-11.0

# new
RUN git clone https://github.com/hughperkins/iverilog.git && \
    cd iverilog && \
    git checkout hp/v11_0_vvp_so && \
    sh autoconf.sh && \
    ./configure && \
    make -j $(nproc) && \
    make install && \
    iverilog -V && \
    cd .. && \
    rm -Rf iverilog

# yosys
RUN wget https://github.com/YosysHQ/yosys/archive/refs/tags/yosys-0.15.tar.gz && \
    tar -xf yosys-0.15.tar.gz && cd yosys-yosys-0.15 && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -Rf yosys-yosys-0.15

# verilator
RUN git clone https://github.com/verilator/verilator && \
    cd verilator && \
    git checkout v4.218 && \
    autoconf && \
    ./configure && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -Rf verilator
