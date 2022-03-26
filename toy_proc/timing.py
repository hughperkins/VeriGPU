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

You should provide --top-module, unless you are provide a top level task.
This can be used on a task, by passing in --top-task, instead of --top-module. There are some constraints
around the formatting of such a task. See header of run_yosys.py for more details on these constraints.


Some details:
- in the gate-level netlist, the only active 'things' are 'cells'
- cells comprise mostly logic gates (NAND etc) and flip-flops (DFFxxx)
- we also have assigns, which we represents as cells with no propagation delay
    - the rhs of the assign is the inputs, and the lhs is the outputs
- the outputs of a DFF are treated as inputs into the module, and the inputs to
  the DFF are outputs from the module
- we measure the longest path between any input and any output, where input means
  any module input, or any dff output; and output means any module output, or any
  dff input

cells data structures:
- cells are stored in cells list
- there are multiple indexes into this list, eg cell_idx_by_input_name, cell_idx_by_cell_name,
  etc
- using these indexes, we can locate a particular cell in O(1) time, using a hashtable lookup
    - so the entire script runs relatively fast (much faster than the synthesis step beforehand)
- the module inputs are represented by a 'START' cell, and the module outputs are represented
  by an 'END' cell

vectors:
- some wires are vectors. They are represented in the netlist sometimes as individual
  wires (foo[2]), and sometimes as entire vectors. We use the original wire declaration
  of each wire to determine whether it's a vector or not. Then, in assigns, we
  expand vectors into the individual wires, where necessary
- the vector declarations are stored in vector_bits_by_name

whitespace:
- sometimes wires have a space between their name and a vector index, eg `foo [2]`
- we strip whitespace as standard
- where whitespace is not stripped, that is a bug
    - typically this will manifest as a bug sooner or later
"""
import argparse
from typing import Dict, Optional, List
from collections import deque, defaultdict, Counter
import subprocess
import sys


g_cell_times = {
    # based loosely on src/timings_ece474.txt
    # taking time for a nand or nor gate as ~1
    'INVX1': 0.6,
    'BUFX1': 1.2,
    'BUFX2': 1.5,
    'BUFX4': 2.0,
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
    'ASSIGN': 0,
}

g_cell_areas = {
    'invx1': 1,
    'invx8': 14.7,
    'and2x1': 1.5,
    'and2x2': 1.5,
    'nand2x1': 1,
    'nand2x2': 2,
    'nand3x1': 2.0,
    'or2x1': 1.5,
    'or2x2': 2.0,
    'nor2x1': 1.0,
    'nor2x2': 2.0,
    'nor3x1': 2.0,
    'xor2x1': 3.0,
    'xor2x2': 3.0,
    'xor3x1': 4.0,
    'xor3x2': 4.0,
    'xnor2x1': 3.0,

    'aoi21x1': 2.0,
    'aio21x2': 2.0,
    'aoi22x1': 2.0,
    'oai21x1': 2.0,
    'oai22x1': 2.0,

    'mux2x1': 2.0,
    'mux21x1': 2.0,
    'mux21x2': 2.0,
    'mux41x1': 4.0,
    'mux41x2': 4.0,

    'dec24x1': 5.0,

    'dffx1': 4.0,
    'dffx2': 5.0,
    'dffposx1': 4.0,
    'dffsr': 6.0,

    'dffnasr1x1': 6.0,
    'dffnasr1x1': 6.0,

    'start': 0,
    'end': 0,
    'assign': 0,
    'unused_bits': 0
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


def str_to_names(vector_bits_by_name: Dict[str, List[int]], names_str: str):
    """
    example of names_str:
    some_name
    some_name[2:15]
    {some_name, another_name[2:15]}

    any of these names can be found in vectors lookup dict, and should then be converted
    into a list of names, using the vector definition
    """
    names = []
    names_str = names_str.replace(' ', '').strip()
    if names_str.startswith('{'):
        # it's a concatenation
        split_names_str = names_str[1:-1].split(',')
        for child_name_str in split_names_str:
            names += str_to_names(vector_bits_by_name, child_name_str.strip())
    elif names_str in vector_bits_by_name:
        # it's a vector, but doesn't say it is
        # use vector_bits_by_name to turn it into a vector
        bits = vector_bits_by_name[names_str]
        names += [f'{names_str}[{i}]' for i in bits]
    elif '[' in names_str and ':' in names_str.split('[')[1]:
        # it's already a vector...
        # split into names
        basename = names_str.split('[')[0].strip()
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
        if args.top_task:
            child_args += ['--top-task', args.top_task]
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

    previous_unused_bits = []

    # each value is a tuple of (start_dim, end_dim_excl, step)
    # that we could feed to range(start, end_excl, step)
    vector_bits_by_name = {}
    for line in netlist.split('\n'):
        line = line.strip()
        next_unused_bits = []
        if in_declaration:
            # inside a cell declaration
            if line.strip() == ');':
                # write out the completed cell declaration
                if cell_type.startswith('DFF'):
                    # # let's treat dff ports as input and output ports of the module
                    cell = Cell(cell_type, cell_name, cell_inputs, cell_outputs, is_source_sink=True)
                    cell_idx = len(cells)
                    cellidx_by_cell_name[cell_name] = cell_idx
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
                port_wires = str_to_names(vector_bits_by_name, port_wire)
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
            elif line.startswith('(* unused_bits'):
                # print('unused bits line', line)
                unused_bits = [int(s) for s in line.split('"')[1].split()]
                # print('unused_bits', unused_bits)
                next_unused_bits = unused_bits
            elif line.startswith('assign'):
                # eg
                # assign a = b;
                line = line.replace('assign ', '').strip()
                line = line[:-1]
                lhs, _, rhs = line.partition(' = ')
                lhs_names = str_to_names(vector_bits_by_name, lhs)
                rhs_names = str_to_names(vector_bits_by_name, rhs)

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
                    vector_bits_by_name[name] = list(range(start, end_excl, step))
                    if len(previous_unused_bits) > 0:
                        # create a source cell that outputs each of the unused bits, like an immediate
                        # for bit in previous_unused_bits:
                        cell = Cell(
                            cell_type='UNUSED_BITS',
                            cell_name=f'{name}_unused_bits',
                            inputs=[],
                            outputs=[f'{name}[{bit}]' for bit in previous_unused_bits],
                            is_source_sink=True
                        )
                        print('unused bits cell', cell)
                        cell.output_delay = 0
                        cell_idx = len(cells)
                        cells.append(cell)
                        source_sink_nodes.add(cell)
                        for cell_output in cell.cell_outputs:
                            cellidxs_by_output[cell_output.strip()].append(cell_idx)
                else:
                    # we dont care abou the vector bit, but if theres an unused bits, then
                    # create an unused bits cell for this wire
                    if len(previous_unused_bits) > 0:
                        _, name = line[:-1].split()
                        assert len(previous_unused_bits) == 1
                        assert previous_unused_bits[0] == 0
                        cell = Cell(
                            cell_type='UNUSED_BITS',
                            cell_name=f'{name}_unused_bit',
                            inputs=[],
                            outputs=[name],
                            is_source_sink=True
                        )
                        print('unused bits cell', cell)
                        cell.output_delay = 0
                        cell_idx = len(cells)
                        cells.append(cell)
                        source_sink_nodes.add(cell)
                        for cell_output in cell.cell_outputs:
                            cellidxs_by_output[cell_output.strip()].append(cell_idx)
            if line.endswith('('):
                in_declaration = True
                cell_type, cell_name, _ = line.split()
                cell_inputs = {}
                cell_outputs = {}
        previous_unused_bits = next_unused_bits

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
                    print('input_name [' + wire_name + '] already in cell_input_delay_by_name')
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
    slowest_node = None
    for node in source_sink_nodes:
        if node.max_input_delay() > max_delay:
            slowest_node = node
            max_delay = node.max_input_delay()

    def walk_node(node, first_node: bool = True):
        slowest_input = None
        # first_node = False
        print(node.cell_type, node.cell_name, '%.1f' % node.cell_delay)
        if not first_node and node.cell_type in ["DFFSR", "START"]:
            print('reached termination')
            return
        longest_delay = -1
        # print('    inputs', node.cell_input_delay_by_name)
        for cell_input, delay in node.cell_input_delay_by_name.items():
            if delay > longest_delay:
                slowest_input = cell_input
                longest_delay = delay
        print('    ', slowest_input, '%.1f' % longest_delay)
        slowest_incomings = cellidxs_by_output[slowest_input]
        # print('slowest_incomings', slowest_incomings)
        assert len(slowest_incomings) == 1
        slowest_incoming = cells[slowest_incomings[0]]
        walk_node(slowest_incoming, first_node=False)

    if args.show_path:
        print('slowest_node', slowest_node)
        walk_node(slowest_node)

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
        sys.exit(1)

    area = 0
    cell_count_by_type = Counter()
    for cell in cells:
        cell_area = g_cell_areas[cell.cell_type.lower()]
        area += cell_area
        cell_count_by_type[cell.cell_type] += 1

    print('')
    if args.show_cell_counts:
        print('Cell counts:')
        cell_infos = []
        for cell_type, count in cell_count_by_type.items():
            cell_area = g_cell_areas[cell_type.lower()]
            cell_infos.append({'cell_type': cell_type, 'count': count, 'area': cell_area * count})
        cell_infos.sort(key=lambda x: x['area'], reverse=True)
        for cell_info in cell_infos:
            if cell_info['area'] == 0:
                continue
            print('    ', cell_info['cell_type'], 'count:', cell_info['count'], ' total area:', cell_info['area'])

    print('')
    print('Propagation delay is between any pair of combinatorially connected')
    print('inputs and outputs, drawn from:')
    print('    - module inputs')
    print('    - module outputs,')
    print('    - flip-flop outputs (treated as inputs), and')
    print('    - flip-flop inputs (treated as outputs)')
    print('')
    print('Max propagation delay: %.1f nand units' % max_delay)
    print('Area:                  %.1f nand units' % area)
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in-netlist', type=str,
        help='path to gate netlist verilog file, choose one of --in-netlist or --in-verilog')
    parser.add_argument('--top-module', type=str, help='top module name, only needed if more than one module.')
    parser.add_argument('--top-task', type=str, help='top task name, if it\'s a task, not a module')
    parser.add_argument(
        '--in-verilog', type=str, nargs='+',
        help='path to original verilog file, choose one of --in-verilog or --in-netlist')
    parser.add_argument(
        '--cell-lib', type=str, default='tech/osu018/osu018_stdcells.lib',
        help='e.g. path to osu018_stdcells.lib')
    parser.add_argument('--show-path', action='store_true', help='print longest path')
    parser.add_argument('--show-cell-counts', action='store_true', help='print count of each cell type')
    args = parser.parse_args()
    if args.in_verilog is not None:
        args.in_verilog = deque(args.in_verilog)
        for additional in ['src/assert_ignore.sv', 'src/op_const.sv', 'src/const.sv']:
            if additional not in args.in_verilog:
                args.in_verilog.appendleft(additional)
        args.in_verilog = list(args.in_verilog)
        print(args.in_verilog)
    # we should have one argument only
    assert args.in_netlist is not None or args.in_verilog is not None
    assert args.in_netlist is None or args.in_verilog is None
    run(args)
