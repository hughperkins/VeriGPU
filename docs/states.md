Brainstorming

Current states
==============

- tick:  # we have to wait for next timestep at least for memory tro response
    pc = 0
    state = C0
    mem_rd_addr = 0
    mem_r_req = 1
    mem_ack: 0
- tick * n:
    pc = 0
    state = C1
    mem_r_req = 0
    mem_ack: 0
- tick
    mem_ack: 1
    pc = 0
    state = C1
    mem_r_data = instr[0]
    c1_instr = instr[0]
    c1_op: correct
    c1_rs1_sel: correct
    c1_rs2_sel: correct
    c1_rsd_sel: correct
    c1_rsd = c1_rs1 op c1_rs2
    next_pc = pc + 4
    next_state = C0
    mem_rd_addr = next_pc
    mem_r_req = 1


Comb states
===========
