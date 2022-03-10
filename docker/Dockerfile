# this is used for cicd

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y gperf flex bison build-essential clang tcl-dev libboost-dev autoconf wget pkg-config \
    libfl-dev zlibc zlib1g zlib1g-dev perl git ccache libgoogle-perftools-dev numactl perl-doc python3 libreadline-dev
RUN python3 -V

# yosys
RUN wget https://github.com/YosysHQ/yosys/archive/refs/tags/yosys-0.15.tar.gz && tar -xf yosys-0.15.tar.gz && cd yosys-yosys-0.15 &&     make -j $(nproc) &&     make install && cd && rm -Rf yosys-yosys-0.15

# iverilog
RUN wget ftp://ftp.icarus.com/pub/eda/verilog//v11/verilog-11.0.tar.gz && tar -xzvf verilog-11.0.tar.gz &&     cd verilog-11.0 && ./configure && make -j $(nproc) && make install && cd && rm -Rf verilog-11.0

# verilator
RUN git clone https://github.com/verilator/verilator && cd verilator && git checkout v4.218 && autoconf && ./configure && make -j $(nproc) && make install && cd && rm -Rf verilator

# python
# RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz &&     tar xzf Python-3.8.5.tgz &&     rm Python-3.8.5.tgz &&     cd Python-3.8.5 &&     ./configure &&     make install

# RUN git clone https://github.com/sstephenson/bats.git   && cd bats   && ./install.sh /usr/local   && cd ..   && rm -rf bats
# RUN perl -MCPAN -e 'install TAP::Parser'
# RUN perl -MCPAN -e 'install XML::Generator'
# RUN apt-get update && apt-get -y install lsb-release unzip
