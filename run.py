import subprocess
import os
import argparse


def run(args):
    assert os.system(f'python assembler.py --in-asm {args.name}.asm --out-hex build/{args.name}.hex') == 0
    with open('comp_driver.sv') as f:
        comp_driver = f.read()
    comp_driver = comp_driver.replace('{PROG}', args.name)
    with open('build/comp_driver.sv', 'w') as f:
        f.write(comp_driver)
    assert os.system('iverilog -g2012 comp.sv mem.sv build/comp_driver.sv') == 0
    os.system('./a.out')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='eg prog5')
    args = parser.parse_args()
    run(args)
