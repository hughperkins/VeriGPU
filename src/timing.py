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
import os
import time
from collections import deque, defaultdict
import networkx as nx
import matplotlib.pyplot as plt


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
    'START': 0,
    'END': 0
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
    with open(args.in_netlist) as f:
        netlist = f.read()
    # print(netlist)
    # inputs = []
    # outputs = []
    cells = []
    input_cell = Cell('START', 'START', [], [], output_delay=0)
    output_cell = Cell('END', 'END', [], [], output_delay=0)
    # input_cell = {
    #     'cell_inputs': [],
    #     'cell_outputs': [],
    #     'cell_type': 'START',
    #     'cell_name': 'START',
    #     'prop_delay_to_outputs': 0,
    # }
    # output_cell = {
    #     'cell_inputs': [],
    #     'cell_outputs': [],
    #     'cell_type': 'END',
    #     'cell_name': 'END',
    #     'prop_delay_to_outputs': None,
    # }
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
    for line in netlist.split('\n'):
        # print(line)
        line = line.strip()
        if in_declaration:
            if line.strip() == ');':
                cell = Cell(cell_type, cell_name, cell_inputs, cell_outputs)
                # cell = {
                #     'cell_inputs': cell_inputs,
                #     'cell_outputs': cell_outputs,
                #     'cell_type': cell_type,
                #     'cell_name': cell_name,
                #     'prop_delay_to_outputs': None,
                # }
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
                if port_name in ['A', 'B', 'C', 'D']:
                    cell_inputs[port_line] = port_line
                else:
                    cell_outputs[port_line] = port_line
        else:
            if line.startswith('input '):
                _, dims, name = line[:-1].split()
                if dims.startswith('['):
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
                        input_cell.cell_outputs.append(f'{name}[{wire}]')
                        cellidx_by_output[f'{name}[{wire}]'] = input_cell_idx

            if line.startswith('output '):
                _, dims, name = line[:-1].split()
                if dims.startswith('['):
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
            if line.endswith('('):
                in_declaration = True
                cell_type, cell_name, _ = line.split()
                cell_inputs = {}
                cell_outputs = {}
    # print('inputs', inputs)
    # print('outputs', outputs)
    # for cell in cells:
        # print(cell)
    print('loaded data from netlist')
    print('')

    G = nx.Graph()
    for i, cell in enumerate(cells):
        G.add_node(cell.cell_name)
    for to_idx, to_cell in enumerate(cells):
        to_name = to_cell.cell_name
        for cell_input in to_cell.cell_inputs:
            # to_idx = i
            from_idx = cellidx_by_output[cell_input]
            from_cell = cells[from_idx]
            from_name = from_cell.cell_name
            # cell_time = g_cell_times[cell.cell_type]
            # output_names = cell.cell_outputs
            # from_idx = cellidx_by_output[output_names[0]]
            G.add_edge(from_name, to_name, name=cell_input)
    # print('G', G)
    # nx.draw(G)
    # plt.show()
    # time.sleep(5)
    nx.nx_pydot.write_dot(G, '/tmp/netlist.dot')
    # os.system('xdot /tmp/netlist.dot')

    # walk graph, starting from inputs
    # we are looking for longest path through the graph
    to_process = deque(input_cell.cell_outputs)
    # for wire_name in to_process:
    while len(to_process) > 0:
        wire_name = to_process.popleft()
        # print('wire_name', wire_name)
        from_idx = cellidx_by_output[wire_name]
        to_idxs = cellidxs_by_input[wire_name]
        for to_idx in to_idxs:
            from_cell = cells[from_idx]
            to_cell = cells[to_idx]
            # print('from', from_cell.cell_name, 'to', to_cell.cell_name)
            delay = from_cell.output_delay
            assert delay is not None
            to_cell.connect_input(wire_name, delay)
            if to_cell.output_delay is not None:
                for wire in to_cell.cell_outputs:
                    to_process.append(wire)

    # for cell in cells:
        # print(cell)
    print('output max delay %.3f' % output_cell.output_delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-netlist', type=str, required=True, help='path to gate netlist verilog file')
    args = parser.parse_args()
    run(args)
