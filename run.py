import os
import argparse
import sys


def run(args):
    args.name = args.name.replace('.asm', '').replace('src/', '')
    assert os.system(
        f'{sys.executable} toy_proc/assembler.py --in-asm examples/{args.name}.asm'
        f' --out-hex build/prog.hex') == 0
    with open('src/comp_driver.sv') as f:
        comp_driver = f.read()
    comp_driver = comp_driver.replace('{PROG}', 'prog')
    with open('build/comp_driver.sv', 'w') as f:
        f.write(comp_driver)
    os.system(f'cat examples/{args.name}.asm')
    assert os.system(
        'iverilog -g2012 -pfileline=1 src/op_const.sv src/const.sv src/int_div_regfile.sv src/proc.sv src/comp.sv'
        ' src/mem_delayed.sv build/comp_driver.sv') == 0
    os.system('./a.out | tee /tmp/out.txt')
    with open('/tmp/out.txt') as f:
        output = f.read()
        if 'ERROR' in output:
            for line in output.split('\n'):
                if 'ERROR' in line:
                    print(line)
            print('')
            raise Exception('Error')
    if os.path.exists(f'examples/{args.name}_expected.txt'):
        with open('/tmp/out.txt') as f:
            output = f.read()
            output = '\n'.join([line for line in output.split('\n') if line.startswith('out')])
        with open(f'examples/{args.name}_expected.txt') as f:
            expected = f.read().strip()
        print('')
        print('Target prog: ' + args.name)
        print('')
        if expected != output:
            print('TEST ERROR: output mismatch')
            print('')
            print('actual:')
            print(output)
            print('')
            print('expected:')
            print(expected)
            raise Exception('assert failed')
        print('output verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='eg prog5')
    args = parser.parse_args()
    run(args)
