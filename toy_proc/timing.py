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
from typing import Dict, Optional, Tuple
from collections import deque, defaultdict
import networkx as nx
import subprocess
import sys


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


INPUT_PORT_NAMES = ['a', 'b', 'c', 'd', 'r', 's']
OUTPUT_PORT_NAMES = ['q', 'y']


class Cell:
    def __init__(self, cell_type, cell_name, inputs, outputs, is_source_sink: bool = False):
        self.cell_type = cell_type
        self.cell_name = cell_name
        self.cell_inputs = inputs
        self.cell_outputs = outputs
        self.cell_input_delay_by_name: Dict[str, float] = {}
        self.is_source_sink = is_source_sink
        self.output_delay: Optional[float] = None
        if is_source_sink:
            self.cell_delay = 1e8
            self.output_delay = 0
        else:
            self.cell_delay = g_cell_times[cell_type]
            self.output_delay = None

    def connect_input(self, input_name: str, delay: float):
        assert input_name not in self.cell_input_delay_by_name
        self.cell_input_delay_by_name[input_name] = delay
        if len(self.cell_input_delay_by_name) == len(self.cell_inputs) and not self.is_source_sink:
            self._calc_output_delay()

    def max_input_delay(self):
        if len(self.cell_input_delay_by_name) == 0:
            max_delay = 0
        else:
            max_delay = max(self.cell_input_delay_by_name.values())
        return max_delay

    def _calc_output_delay(self):
        self.output_delay = max(self.cell_input_delay_by_name.values()) + self.cell_delay

    def __repr__(self):
        return (
            f'Cell(type={self.cell_type} name={self.cell_name} output_delay={self.output_delay}'
            f' in={len(self.cell_input_delay_by_name)}/{len(self.cell_inputs)})')


def str_to_names(vector_dims_by_name: Dict[str, Tuple[int, int, int]], names_str: str):
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
            names += str_to_names(vector_dims_by_name, child_name_str.strip())
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
        names.append(names_str.strip())
    names = sorted(list(set(names)))
    return names


def run(args):
    if args.in_verilog is not None:
        # first need to synthesize
        # use check output, so we can suppress output (less spammy...)
        child_args = [
            sys.executable, 'toy_proc/run_yosys.py', '--in-verilog'] + args.in_verilog + [
            '--cell-lib', args.cell_lib
        ]
        if args.top_module:
            child_args += ['--top-module', args.top_module]
        subprocess.check_output(child_args)
        args.in_netlist = 'build/netlist.v'

    with open(args.in_netlist) as f:
        netlist = f.read()
    input_wires = set()  # from the lines starting with 'input ', at the top
    output_wires = set()  # form the lines starting with 'output ' at the top
    cells = []
    start_cell = Cell('START', 'START', [], [], is_source_sink=True)
    end_cell = Cell('END', 'END', [], [], is_source_sink=True)
    cells.append(start_cell)
    cells.append(end_cell)
    start_cell_idx = 0
    end_cell_idx = 1
    cell_inputs = {}
    cell_outputs = {}
    cellidxs_by_input = defaultdict(list)  # outputs fo the module are inputs to the END module, so appear here
    cellidxs_by_output = defaultdict(list)  # inputs to the module are outputs of the START cell, so appear here
    source_sink_nodes = set()  # START, END and DFF nodes. This contains the source and sink nodes
    source_sink_nodes.add(start_cell)
    source_sink_nodes.add(end_cell)
    cellidx_by_cell_name = {}
    in_declaration = False
    cell_type = ''
    cell_name = ''
    assign_idx = 0  # 0-indexed
    # for empty cells that we should walk from straight away, since we know their
    # output_delay is 0
    empty_cells = []

    # each value is a tuple of (start_dim, end_dim_excl, step)
    # that we could feed to range(start, end_excl, step)
    vector_dims_by_name = {}
    for line in netlist.split('\n'):
        line = line.strip()
        if in_declaration:
            # inside a cell declaration
            if line.strip() == ');':
                # write out the completed cell declaration
                if cell_type.startswith('DFF'):
                    # # let's treat dff ports as input and output ports of the module
                    cell = Cell(cell_type, cell_name, cell_inputs, cell_outputs, is_source_sink=True)
                    cell_idx = len(cells)
                    cells.append(cell)
                    source_sink_nodes.add(cell)
                    for cell_input in cell_inputs.keys():
                        cellidxs_by_input[cell_input].append(cell_idx)
                    for cell_output in cell_outputs.keys():
                        cellidxs_by_output[cell_output.strip()].append(cell_idx)
                else:
                    cell = Cell(cell_type, cell_name, cell_inputs, cell_outputs)
                    cell_idx = len(cells)
                    cells.append(cell)
                    cellidx_by_cell_name[cell_name] = cell_idx
                    for cell_input in cell_inputs.keys():
                        cellidxs_by_input[cell_input].append(cell_idx)
                    for cell_output in cell_outputs.keys():
                        cellidxs_by_output[cell_output.strip()].append(cell_idx)
                in_declaration = False
            else:
                port_name = line.split('.')[1].split('(')[0].lower()
                port_wire = line.split('(')[1].split(')')[0].strip().replace(' ', '')
                # ignore immediate numbers
                if port_wire[0] in '0123456789':
                    continue
                port_wires = str_to_names(vector_dims_by_name, port_wire)
                for port_wire in port_wires:
                    if port_name in input_wires:
                        cell_inputs[port_wire] = port_wire
                    elif port_name in output_wires:
                        cell_outputs[port_wire] = port_wire
                    elif port_name in INPUT_PORT_NAMES:
                        cell_inputs[port_wire] = port_wire
                    elif port_name in OUTPUT_PORT_NAMES:
                        cell_outputs[port_wire] = port_wire
                    else:
                        raise Exception('unknown port name', port_name)
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
                name = name.strip()
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
                        start_cell.cell_outputs.append(f'{name}[{wire}]')
                        cellidxs_by_output[f'{name}[{wire}]'.strip()].append(start_cell_idx)
                        input_wires.add(f'{name}[{wire}]')
                        # print('input [' + f'{name}[{wire}]' + ']')
                else:
                    start_cell.cell_outputs.append(f'{name}')
                    cellidxs_by_output[f'{name}'].append(start_cell_idx)
                    input_wires.add(name)
                    # print('input [' + f'{name}' + ']')
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
                            end_cell.cell_inputs.append(f'{name}[{wire}]')
                            cellidxs_by_input[f'{name}[{wire}]'].append(end_cell_idx)
                            output_wires.add(f'{name}[{wire}]')
                    else:
                        end_cell.cell_inputs.append(f'{name}')
                        cellidxs_by_input[f'{name}'].append(end_cell_idx)
                        output_wires.add(name)
                except Exception as e:
                    print(line)
                    raise e
            elif line.startswith('assign'):
                # eg
                # assign a = b;
                line = line.replace('assign ', '').strip()
                line = line[:-1]
                lhs, _, rhs = line.partition(' = ')
                lhs_names = str_to_names(vector_dims_by_name, lhs)
                rhs_names = str_to_names(vector_dims_by_name, rhs)

                cell_outputs = lhs_names
                cell_inputs = rhs_names

                output_delay = None
                if len(cell_inputs) == 0:
                    output_delay = 0
                cell = Cell(
                    cell_type='ASSIGN',
                    cell_name='assign' + str(assign_idx),
                    inputs=cell_inputs,
                    outputs=cell_outputs,
                )
                cell.output_delay = output_delay
                assign_idx += 1
                cell_idx = len(cells)
                cells.append(cell)
                for cell_input in cell_inputs:
                    cellidxs_by_input[cell_input.strip()].append(cell_idx)
                for cell_output in cell_outputs:
                    cellidxs_by_output[cell_output.strip()].append(cell_idx)
                if len(cell_inputs) == 0:
                    empty_cells += cell_outputs
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

    G = nx.Graph()
    for i, cell in enumerate(cells):
        G.add_node(cell.cell_name)
    for to_idx, to_cell in enumerate(cells):
        to_name = to_cell.cell_name
        for cell_input in to_cell.cell_inputs:
            try:
                from_idxs = cellidxs_by_output[cell_input]
            except Exception as e:
                print('failed to find ' + cell_input + ' in cellidx_by_output')
                print('cellidx_by_output keys:')
                # for wire in cellidx_by_output.keys():
                #     print('- wire [' + wire + ']')
                print('')
                print('to_idx', to_idx, to_cell.cell_name, to_cell, 'cell_input', cell_input)
                raise e
            for from_idx in from_idxs:
                from_cell = cells[from_idx]
                from_name = from_cell.cell_name
                G.add_edge(from_name, to_name, name=cell_input)
    nx.nx_pydot.write_dot(G, 'build/netlist.dot')

    # walk graph, starting from inputs
    # we are looking for longest path through the graph
    # to_process = deque(start_cell.cell_outputs)
    to_process = deque()
    for source_sink_cell in source_sink_nodes:
        to_process.extend(source_sink_cell.cell_outputs)
    seen = set(to_process)
    to_process += empty_cells
    while len(to_process) > 0:
        wire_name = to_process.popleft()
        from_idxs = cellidxs_by_output[wire_name]
        to_idxs = cellidxs_by_input[wire_name]
        for from_idx in from_idxs:
            for to_idx in to_idxs:
                from_cell = cells[from_idx]
                to_cell = cells[to_idx]
                delay = from_cell.output_delay
                assert delay is not None
                try:
                    to_cell.connect_input(wire_name, delay)
                except Exception as e:
                    print('input_name ' + wire_name + ' already in cell_input_delay_by_name')
                    print('cell_input_delay_by_name:')
                    for key, value in to_cell.cell_input_delay_by_name.items():
                        print('-', key, value)
                    print('')
                    print(
                        'from cell name', from_cell.cell_name, 'wire name', wire_name,
                        'to cell name', to_cell.cell_name, 'delay', delay)
                    print('to_cell inputs')
                    for _ in to_cell.cell_inputs:
                        print('    ', _)
                    print('to_cell outputs')
                    for _ in to_cell.cell_outputs:
                        print('    ', _)
                    raise e
                if to_cell.output_delay is not None:
                    for wire in to_cell.cell_outputs:
                        if wire not in seen:
                            to_process.append(wire)
                            seen.add(wire)

    max_delay = 0
    for node in source_sink_nodes:
        max_delay = max(max_delay, node.max_input_delay())

    # check for unprocessed nodes
    printed_prologue = False
    for cell in cells:
        if cell.output_delay is None:
            if not printed_prologue:
                print('output delay not known:')
                printed_prologue = True
            print(cell)
            for name in cell.cell_inputs:
                if name not in cell.cell_input_delay_by_name:
                    print('    missing', name)
    if printed_prologue:
        return

    print('')
    print('Propagation delay is between any pair of combinatorially connected')
    print('inputs and outputs, drawn from:')
    print('    - module inputs')
    print('    - module outputs,')
    print('    - flip-flop outputs (treated as inputs), and')
    print('    - flip-flop inputs (treated as outputs)')
    print('')
    print('max propagation delay: %.1f nand units' % max_delay)
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-netlist', type=str, help='path to gate netlist verilog file')
    parser.add_argument('--top-module', type=str, help='top module name, only needed if more than one module.')
    parser.add_argument('--in-verilog', type=str, nargs='+', help='path to original verilog file')
    parser.add_argument(
        '--cell-lib', type=str, default='tech/osu018/osu018_stdcells.lib',
        help='e.g. path to osu018_stdcells.lib')
    args = parser.parse_args()
    # we should have one argument only
    assert args.in_netlist is not None or args.in_verilog is not None
    assert args.in_netlist is None or args.in_verilog is None
    run(args)
