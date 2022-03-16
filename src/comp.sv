// contains proc.sv, and any memory elements, such as cache, dummy memory controller, etc
// this will be slower to syntehsize than proc.sv, because of the memory elements
module comp(
    input clk,
    input rst,

    input [31:0] oob_wr_addr,
    input [31:0] oob_wr_data,
    input oob_wen,

    output mem_rd_req, mem_wr_req,
    output mem_busy,
    output mem_ack,

    output [31:0] out,
    output outen,
    output outflen,

    output [6:0] op,
    output [4:0] rd,
    output [6:0] imm1,
    output [31:0] x1,
    output [31:0] pc,
    output [4:0] state,
    output halt
);
    reg [31:0] mem_addr;
    reg [31:0] mem_rd_data, mem_wr_data;

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
        .rst(rst), .clk(clk), .out(out),
        // .c2_op(op), .c2_imm1(imm1),
        .pc(pc),
        .outflen(outflen),
        // .c2_rd_sel(rd),
        .x1(x1),
        .state(state), .outen(outen), .halt(halt),

        .mem_addr(mem_addr),
        .mem_rd_data(mem_rd_data), .mem_wr_data(mem_wr_data),
        .mem_ack(mem_ack), .mem_busy(mem_busy),
        .mem_rd_req(mem_rd_req), .mem_wr_req(mem_wr_req)
    );

endmodule
