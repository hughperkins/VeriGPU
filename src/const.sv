parameter num_regs = 32;

parameter data_width = 32;
parameter addr_width = 32;
parameter instr_width = 32;

// parameter num_regs = 4;
// parameter data_width = 32;
// parameter addr_width = 32;

// parameter num_regs = 1;
// parameter data_width = 10;
// parameter addr_width = 1;
// parameter instr_width = 10;// parameter op_width = 10;

parameter reg_sel_width = $clog2(num_regs);
