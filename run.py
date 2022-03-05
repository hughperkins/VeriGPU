import os
import argparse


def run(args):
    assert os.system(f'python src/assembler.py --in-asm examples/{args.name}.asm --out-hex build/{args.name}.hex') == 0
    with open('src/comp_driver.sv') as f:
        comp_driver = f.read()
    comp_driver = comp_driver.replace('{PROG}', args.name)
    with open('build/comp_driver.sv', 'w') as f:
        f.write(comp_driver)
    os.system(f'cat examples/{args.name}.asm')
    # mem_mod = 'src/mem_delayed.sv' if args.delayed_mem else 'src/mem.sv'
    assert os.system('iverilog -g2012 src/proc.sv src/comp.sv src/mem_delayed.sv build/comp_driver.sv') == 0
    os.system('./a.out | tee /tmp/out.txt')
    if os.path.exists(f'examples/{args.name}_expected.txt'):
        with open('/tmp/out.txt') as f:
            output = f.read()
            output = '\n'.join([line for line in output.split('\n') if line.startswith('out')])
        # print('')
        # print(output)
        with open(f'examples/{args.name}_expected.txt') as f:
            expected = f.read().strip()
        if expected != output:
            print('output mismatch')
            print(output)
            print('')
            print(expected)
            raise Exception('assert failed')
        print('output verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='eg prog5')
    # parser.add_argument('--delayed-mem', action='store_true', help='use delayed memory')
    args = parser.parse_args()
    run(args)
