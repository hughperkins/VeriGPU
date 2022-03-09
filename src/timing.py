"""
Take a gate-level netlist .v file, from yosys, and run depth first search, to
find the longest path, given certain inputs, through combinatorial logic.

Will use 'unit times' for timings, which will look something like:

- nand: 1
- nor: 1
- not: 1
- and: 2 (nand + not)
- or: 2 (nor + nor)
- xor: 3 (various constructions...)
- xnor: 3 (various constructions...)

- shl: 0 (it's just rewiring)
- shr: 0 (it's just rewiring)
"""
import argparse
from collections import deque, defaultdict
import networkx as nx
import subprocess


g_cell_times = {
    # based loosely on src/timings_ece474.txt
    # taking time for a nand or nor gate as ~1
    'INVX1': 0.6,
    'NAND2X1': 1,
    'NAND3X1': 2.6,
    'NOR2X1': 1,
    'XOR2X1': 2.6,
    'XNOR2X1': 2.6,
    'NOR3X1': 2,
    'AND2X1': 1.6,
    'OAI21X1': 2,
    'AOI21X1': 2.6,
    'AOI22X1': 3,
    'OAI22X1': 3,
    'OR2X1': 1.6,
    'MUX2X1': 2,
    'MUX4X1': 3,
    'START': 0,
    'END': 0,
    'ASSIGN': 0
}


class Cell:
    def __init__(self, cell_type, cell_name, inputs, outputs, output_delay: float = None):
        self.cell_type = cell_type
        self.cell_name = cell_name
        self.cell_inputs = inputs
        self.cell_outputs = outputs
        self.cell_input_delay_by_name = {}
        self.cell_delay = g_cell_times[cell_type]
        self.output_delay = output_delay

    def connect_input(self, input_name: str, delay: float):
        assert input_name not in self.cell_input_delay_by_name
        self.cell_input_delay_by_name[input_name] = delay
        if len(self.cell_input_delay_by_name) == len(self.cell_inputs):
            self._calc_output_delay()

    def _calc_output_delay(self):
        self.output_delay = max(self.cell_input_delay_by_name.values()) + self.cell_delay

    def __repr__(self):
        return (
            f'Cell({self.cell_type} {self.cell_name} output_delay={self.output_delay}'
            f' in={len(self.cell_input_delay_by_name)}/{len(self.cell_inputs)})')


def run(args):
    def str_to_names(names_str: str):
        """
        example of names_str:
        some_name
        some_name[2:15]
        {some_name, another_name[2:15]}

        any of these names can be found in vectors lookup dict, and should then be converted
        into a list of names, using the vector definition
        """
        names = []
        names_str = names_str.strip()
        if names_str.startswith('{'):
            # it's a concatenation
            split_names_str = names_str[1:-1].split(',')
            for child_name_str in split_names_str:
                names += str_to_names(child_name_str)
        elif names_str in vector_dims_by_name:
            start, end, step = vector_dims_by_name[names_str]
            names += [f'{names_str}[{i}]' for i in range(start, end, step)]
        elif ':' in names_str:
            # it's already a vector...
            # split into names
            basename = names_str.split('[')[0]
            start = int(names_str.split('[')[1].split(':')[0])
            end = int(names_str.split(':')[1].split(']')[0])
            step = 1 if end >= start else -1
            end_excl = end + step
            names += [f'{basename}[{i}]' for i in range(start, end_excl, step)]
        elif "'" in names_str:
            # immediate constant, ignore
            pass
        else:
            names.append(names_str)
        return list(set(names))
        return names

    if args.in_verilog is not None:
        # first need to synthesize
        # use check output, so we can suppress output (less spammy...)
        subprocess.check_output([
            'python', 'src/tools/run_yosys.py', '--verilog', args.in_verilog
        ])
        # os.system(f'python src/tools/run_yosys.py --verilog {args.in_verilog}')
        args.in_netlist = 'build/netlist.v'

    with open(args.in_netlist) as f:
        netlist = f.read()
    cells = []
    input_cell = Cell('START', 'START', [], [], output_delay=0)
    output_cell = Cell('END', 'END', [], [])
    cells.append(input_cell)
    cells.append(output_cell)
    input_cell_idx = 0
    output_cell_idx = 1
    cell_inputs = {}
    cell_outputs = {}
    cellidxs_by_input = defaultdict(list)
    cellidx_by_output = {}
    cellidx_by_cell_name = {}
    in_declaration = False
    cell_type = ''
    cell_name = ''
    # each value is a tuple of (start_dim, end_dim_excl, step)
    # that we could feed to range(start, end_excl, step)
    vector_dims_by_name = {}
    for line in netlist.split('\n'):
        line = line.strip()
        if in_declaration:
            if line.strip() == ');':
                if cell_type.startswith('DFF'):
                    # let's treat dff ports as input and output ports of the module
                    for cell_input in cell_inputs.keys():
                        output_cell.cell_inputs.append(cell_input)
                        cellidxs_by_input[cell_input].append(output_cell_idx)
                    for cell_output in cell_outputs.keys():
                        input_cell.cell_outputs.append(cell_output)
                        cellidx_by_output[cell_output] = input_cell_idx
                else:
                    cell = Cell(cell_type, cell_name, cell_inputs, cell_outputs)
                    cell_idx = len(cells)
                    cells.append(cell)
                    cellidx_by_cell_name[cell_name] = cell_idx
                    for cell_input in cell_inputs.keys():
                        cellidxs_by_input[cell_input].append(cell_idx)
                    for cell_output in cell_outputs.keys():
                        cellidx_by_output[cell_output] = cell_idx
                in_declaration = False
            else:
                port_name = line.split('.')[1].split('(')[0]
                port_line = line.split('(')[1].split(')')[0]
                # ignore immediate numbers
                if port_line[0] in '0123456789':
                    continue
                # yosys sometimes puts spaces between diensions. remove these
                port_line = port_line.replace('] [', '][')
                if port_name in ['A', 'B', 'C', 'D', 'R', 'S']:
                    cell_inputs[port_line] = port_line
                else:
                    cell_outputs[port_line] = port_line
        else:
            if line.startswith('input '):
                try:
                    split_line = line[:-1].split()
                    if len(split_line) == 3:
                        _, dims, name = split_line
                    else:
                        _, name = split_line
                        dims = None
                except Exception as e:
                    print('line ', line)
                    raise e
                if dims is not None and dims.startswith('['):
                    start = int(dims.split('[')[1].split(':')[0])
                    end = int(dims.split(':')[1].split(']')[0])
                    step = 1
                    if end < start:
                        step = -1
                        end -= 1
                    else:
                        end += 1
                    for wire in range(start, end, step):
                        input_cell.cell_outputs.append(f'{name}[{wire}]')
                        cellidx_by_output[f'{name}[{wire}]'] = input_cell_idx
                else:
                    input_cell.cell_outputs.append(f'{name}')
                    cellidx_by_output[f'{name}'] = input_cell_idx
            if line.startswith('output '):
                try:
                    split_line = line[:-1].split()
                    if len(split_line) == 3:
                        _, dims, name = split_line
                    else:
                        _, name = split_line
                        dims = None
                    if dims is not None and dims.startswith('['):
                        start = int(dims.split('[')[1].split(':')[0])
                        end = int(dims.split(':')[1].split(']')[0])
                        # print('start', start, 'end', end)
                        step = 1
                        if end < start:
                            step = -1
                            end -= 1
                        else:
                            end += 1
                        for wire in range(start, end, step):
                            output_cell.cell_inputs.append(f'{name}[{wire}]')
                            cellidxs_by_input[f'{name}[{wire}]'].append(output_cell_idx)
                    else:
                        output_cell.cell_inputs.append(f'{name}')
                        cellidxs_by_input[f'{name}'].append(output_cell_idx)
                except Exception as e:
                    print(line)
                    raise e
            elif line.startswith('assign'):
                # eg
                # assign a = b;
                line = line.replace('assign ', '').strip()
                line = line[:-1]
                lhs, _, rhs = line.partition(' = ')
                lhs_names = str_to_names(lhs)
                rhs_names = str_to_names(rhs)

                cell_outputs = lhs_names
                cell_inputs = rhs_names

                cell = Cell(
                    cell_type='ASSIGN',
                    cell_name='assign' + str(len(cells)),
                    inputs=cell_inputs,
                    outputs=cell_outputs
                )
                cell_idx = len(cells)
                cells.append(cell)
                for cell_input in cell_inputs:
                    cellidxs_by_input[cell_input].append(cell_idx)
                for cell_output in cell_outputs:
                    cellidx_by_output[cell_output] = cell_idx
            elif line.startswith('wire'):
                if '[' in line:
                    _, dims, name = line[:-1].split()
                    start = int(dims.split('[')[1].split(':')[0])
                    end = int(dims.split(':')[1].split(']')[0])
                    step = 1 if end > start else -1
                    end_excl = end + step
                    vector_dims_by_name[name] = (start, end_excl, step)
            if line.endswith('('):
                in_declaration = True
                cell_type, cell_name, _ = line.split()
                cell_inputs = {}
                cell_outputs = {}

    print('vectors:')
    for name, dims in vector_dims_by_name.items():
        print('    ', name, dims)

    G = nx.Graph()
    for i, cell in enumerate(cells):
        G.add_node(cell.cell_name)
    for to_idx, to_cell in enumerate(cells):
        to_name = to_cell.cell_name
        for cell_input in to_cell.cell_inputs:
            try:
                from_idx = cellidx_by_output[cell_input]
            except Exception as e:
                for wire in cellidx_by_output.keys():
                    print('wire', wire)
                print('to_idx', to_idx, to_cell.cell_name, to_cell, 'cell_input', cell_input)
                raise e
            from_cell = cells[from_idx]
            from_name = from_cell.cell_name
            G.add_edge(from_name, to_name, name=cell_input)
    nx.nx_pydot.write_dot(G, '/tmp/netlist.dot')

    # walk graph, starting from inputs
    # we are looking for longest path through the graph
    to_process = deque(input_cell.cell_outputs)
    while len(to_process) > 0:
        wire_name = to_process.popleft()
        from_idx = cellidx_by_output[wire_name]
        to_idxs = cellidxs_by_input[wire_name]
        for to_idx in to_idxs:
            from_cell = cells[from_idx]
            to_cell = cells[to_idx]
            delay = from_cell.output_delay
            assert delay is not None
            try:
                to_cell.connect_input(wire_name, delay)
            except Exception as e:
                print(from_cell.cell_name, wire_name, to_cell.cell_name, delay)
                print('to_cell inputs')
                for _ in to_cell.cell_inputs:
                    print('    ', _)
                print('to_cell outputs')
                for _ in to_cell.cell_outputs:
                    print('    ', _)
                raise e
            if to_cell.output_delay is not None:
                for wire in to_cell.cell_outputs:
                    to_process.append(wire)

    print('')
    print('Propagation delay is between any pair of combinatorially connected')
    print('inputs and outputs, drawn from:')
    print('    - module inputs')
    print('    - module outputs,')
    print('    - flip-flop outputs (treated as inputs), and')
    print('    - flip-flop inputs (treated as outputs)')
    print('')
    print('max propagation delay: %.1f nand units' % output_cell.output_delay)
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-netlist', type=str, help='path to gate netlist verilog file')
    parser.add_argument('--in-verilog', type=str, help='path to original verilog file')
    args = parser.parse_args()
    # we should have one argument only
    assert args.in_netlist is not None or args.in_verilog is not None
    assert args.in_netlist is None or args.in_verilog is None
    run(args)
