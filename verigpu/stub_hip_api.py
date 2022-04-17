"""
Repeatedly tries to run hip pytorch, see what symbol is missing, then add it to build/hip_api.cpp.
Rinse and repeat.

Much faster and easier than doing this by hand :D
"""
import subprocess
import argparse
import os


def names_to_api(names_file, api_file):
    with open(names_file) as f:
        names = [line.strip() for line in f.read().split('\n') if line.strip() != '']
    with open(api_file, 'w') as f:
        f.write("""// initially just create stubs for everything.

#include <iostream>

extern "C" {
""")
        for name in names:
            f.write(f"""    void {name}()
    {{
        std::cout << "{name}" << std::endl;
    }}
""")
        f.write("}\n")


def run(args):
    while True:
        names_to_api(args.name_list_file, args.hip_api_file)
        assert not os.system('src/gpu_runtime/build-cmake.sh')
        if not os.system('python test.py 2>&1 | tee build/test-out.txt'):
            print('failed')
            with open('build/test-out.txt') as f:
                output = f.read()
                print('output', output)
            assert 'undefined symbol' in output
            symbol = output.split('undefined symbol: ')[1].split(',')[0].split(' ')[0].strip()
            print('symbol', symbol)
            cmd = f'echo {symbol} >> build/hip_symbol_list.txt'
            print(cmd)
            assert not os.system(cmd)
        else:
            print('ran ok')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name-list-file', type=str, default='build/hip_symbol_list.txt')
    parser.add_argument('--hip-api-file', type=str, default='build/hip_api.cpp')
    args = parser.parse_args()
    run(args)
