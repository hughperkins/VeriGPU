# this is used for cicd

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y gperf flex bison build-essential clang tcl-dev libboost-dev autoconf wget pkg-config \
    libfl-dev zlibc zlib1g zlib1g-dev perl git ccache libgoogle-perftools-dev numactl \
    perl-doc python3 python3-pip python-is-python3 libreadline-dev cmake xz-utils expect
RUN python3 -V

COPY docker_files docker_files

# clang and llvm 13
RUN wget https://apt.llvm.org/llvm-snapshot.gpg.key && \
    apt-key add llvm-snapshot.gpg.key && \
    cp -v docker_files/llvm-sources.list /etc/apt/sources.list.d/ && \
    apt-get update && \
    apt-get install -y clang-13 lldb-13 lld-13 libllvm13 llvm-13 llvm-13-dev llvm-13-runtime && \
    clang++-13 --version

# llvm 14.0.0
# RUN wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
#     tar -xf clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
#     chown -R root:root clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04 && \
#     mkdir -p /usr/local/opt && \
#     mv clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04 /usr/local/opt/llvm-14.0.0 && \
#     rm clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
#     /usr/local/opt/llvm-14.0.0/bin/clang++ --version

# iverilog 11.0
RUN wget ftp://ftp.icarus.com/pub/eda/verilog//v11/verilog-11.0.tar.gz && \
    tar -xzvf verilog-11.0.tar.gz && \
    cd verilog-11.0 && \
    ./configure && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -Rf verilog-11.0

# from git
# RUN git clone https://github.com/hughperkins/iverilog.git && \
#     cd iverilog && \
#     git checkout hp/v11_0_vvp_so && \
#     sh autoconf.sh && \
#     ./configure && \
#     make -j $(nproc) && \
#     make install && \
#     iverilog -V && \
#     cd .. && \
#     rm -Rf iverilog

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
