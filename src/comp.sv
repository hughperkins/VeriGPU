// represents a computer, i.e. the combination of processor (proc) and
// memory controller (mem)
module comp(
    input clk,
    input rst,

    input [15:0] oob_wr_addr,
    input [15:0] oob_wr_data,
    input oob_wen,

    output mem_d_req, mem_wr_req,
    output mem_busy,
    output mem_ack,

    output [15:0] out,
    output [3:0] op,
    output [3:0] reg_select,
    output [7:0] p1,
    output [7:0] x1,
    output [15:0] pc,
    output [4:0] state,
    output outen
);
    reg [15:0] mem_addr;
    reg [15:0] mem_rd_data, mem_wr_data;

    mem_delayed mem1(
        .clk(clk),

        .addr(mem_addr),
        .wr_req(mem_wr_req), .rd_req(mem_rd_req),
        .rd_data(mem_rd_data), .wr_data(mem_wr_data),
        .busy(mem_busy), .ack(mem_ack),

        .oob_wr_addr(oob_wr_addr), .oob_wr_data(oob_wr_data),
        .oob_wen(oob_wen)
    );

    proc proc1(
        .rst(rst), .clk(clk), .out(out), .op(op), .p1(p1), .pc(pc),
        .reg_select(reg_select),
        .x1(x1),
        .state(state), .outen(outen),

        .mem_addr(mem_addr),
        .mem_rd_data(mem_rd_data), .mem_wr_data(mem_wr_data),
        .mem_ack(mem_ack), .mem_busy(mem_busy),
        .mem_rd_req(mem_rd_req), .mem_wr_req(mem_wr_req)
        //.mem_we(mem_we)
    );

endmodule
