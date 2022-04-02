# prot2

This prototype is for using c++ as the main driver program, which then loads libvvp.so, starts a simulation, and the simulation calls back into the same executable.

In order to run this you need to use the branch of iverilog at https://github.com/hughperkins/iverilog/tree/hp/vvp-so

## Setup/installation

```
# on linux
sudo apt-get install clang bison gperf flex build-essential libreadline-dev
# (or redhat equivalent)

# on Mac
brew install bison
export PATH="/usr/local/opt/bison/bin:$PATH"

bison -V
# bison must be at least version 3

# if you already used a package manager to install iverilog, uinstall it, eg on Mac
# brew uninstall icarus-verilog

git clone https://github.com/hughperkins/iverilog.git
cd iverilog
git checkout hp/v11_0_vvp_so

sh autoconf.sh
./configure
make clean
make -j $(nproc)

# if you're on a Mac, with /usr/local writable:
make install

# otherwise, linux etc:
sudo make install
```

## To run

```
prot/vpi/prot2/run.sh
```

Expected output:
```
++ dirname ../run.sh
+ scriptdir=..
+ echo scriptdir ..
scriptdir ..
+ cd ..
+ [[ ! -e build ]]
+ cd build
+ iverilog -g2012 -ohello.vvp ../hello.sv
../hello.sv:21: warning: @* found no sensitivities so it will never trigger.
+ g++ -c ../test_client.cpp
+ g++ -o test_client test_client.o -lvvp
+ ./test_client
Warning: vvp input file may not be correct version!
i 0 111
i 1 321
i 2 444
i 3 x
i 4 x
i 5 x
i 6 x
i 7 x
i 8 x
i 9 x
i 10 x
i 11 x
i 12 x
i 13 x
i 14 x
i 15 x
```
