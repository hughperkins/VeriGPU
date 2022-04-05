"""
Run each of the examples, and output their timings, and total time, in clock cycles

relies on comp_driver.sv outputing a line starting with 'clock_cycles' (lower case)

similar to reg_test.sh, but reg_test.sh is just checking everything runs; this is reporting
the timings. It uses subprocess.check_output, so you cant see the runs in progress
"""
import argparse
import subprocess
import sys
import glob


# progs = [f'prog{i}' for i in range(2, 23)]
# progs=$(ls -b examples/*.sm)


def run(args):
    progs = glob.glob('examples/direct/*.asm')
    print('progs', progs)

    cycle_counts = []
    for prog in progs:
        output = subprocess.check_output([
            sys.executable, 'run.py', '--name', prog], stderr=subprocess.STDOUT).decode('utf-8')
        # print(output)
        # print(prog)
        cycle_count = int(''.join(
            [line for line in output.split('\n') if line.startswith('cycle_count ')]).split(' ')[1])
        print(prog, 'cycle_count', cycle_count)
        cycle_counts.append(cycle_count)
    print('cycle count is number of clock cycles from reset going low, to halt received.')
    print('')
    print('total', sum(cycle_counts))
    print('avg %.1f' % (sum(cycle_counts) / len(cycle_counts)))
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(args)
