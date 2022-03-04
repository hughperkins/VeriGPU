// represents a computer, i.e. the combination of processor (proc) and
// memory controller (mem)
module comp(
    input clk,
    input rst,

    input [15:0] oob_write_addr,
    input [15:0] oob_write_data,
    input oob_mem_wen,

    output [15:0] out,
    output [3:0] op,
    output [3:0] reg_select,
    output [7:0] p1,
    output [7:0] x1,
    output [15:0] pc,
    output [4:0] state,
    output outen
);
    wire mem_we;

    reg [15:0] mem_read_addr, mem_write_addr;
    reg [15:0] mem_read_data, mem_write_data;

    mem mem1(
        .clk(clk), .we(mem_we),

        .read_addr(mem_read_addr), .write_addr(mem_write_addr),
        .read_data(mem_read_data), .write_data(mem_write_data),

        .oob_write_addr(oob_write_addr), .oob_write_data(oob_write_data),
        .oob_wen(oob_mem_wen)
    );

    proc proc1(
        .rst(rst), .clk(clk), .out(out), .op(op), .p1(p1), .pc(pc),
        .reg_select(reg_select),
        .x1(x1),
        .state(state), .outen(outen),
        .mem_read_addr(mem_read_addr), .mem_write_addr(mem_write_addr),
        .mem_read_data(mem_read_data), .mem_write_data(mem_write_data),
        .mem_we(mem_we)
    );

endmodule
