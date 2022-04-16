// `timescale 1ns/10ps

// places a single compute_unit with a global emmroy, so we can test the computer unit
// in isolation, without gpu controller etc

module compute_unit_and_mem(
    input clk, rst,
    input contr_mem_wr_en,
    input [addr_width - 1:0] contr_mem_wr_addr,
    input [data_width - 1:0] contr_mem_wr_data,

    output reg [data_width - 1:0] out,
    output reg outen,
    output reg outflen,

    input contr_core1_ena,
    input contr_core1_clr,
    input contr_core1_set_pc_req,
    input [data_width - 1:0] contr_core1_set_pc_addr,
    output reg contr_core1_halt
);

    // wire [31:0] out;
    // wire outen;
    // wire outflen;

    wire core1_mem_rd_req;
    wire core1_mem_wr_req;

    wire [addr_width - 1:0] core1_mem_addr;
    wire [data_width - 1:0] core1_mem_rd_data;
    wire [data_width - 1:0] core1_mem_wr_data;

    wire core1_mem_busy;
    wire core1_mem_ack;

    reg contr_mem_rd_en;
    reg [addr_width - 1:0] contr_mem_rd_addr;
    reg [data_width - 1:0] contr_mem_rd_data;
    reg contr_mem_rd_ack;


    // reg [63:0] double;

    global_mem_controller global_mem_controller_(
        .clk(clk),
        .rst(rst),

        .core1_addr(core1_mem_addr),
        .core1_wr_req(core1_mem_wr_req),
        .core1_rd_req(core1_mem_rd_req),
        .core1_rd_data(core1_mem_rd_data),
        .core1_wr_data(core1_mem_wr_data),
        .core1_busy(core1_mem_busy),
        .core1_ack(core1_mem_ack),

        .contr_wr_en(contr_mem_wr_en),
        .contr_rd_en(contr_mem_rd_en),
        .contr_wr_addr(contr_mem_wr_addr),
        .contr_wr_data(contr_mem_wr_data),
        .contr_rd_addr(contr_mem_rd_addr),
        .contr_rd_data(contr_mem_rd_data),
        .contr_rd_ack(contr_mem_rd_ack)
    );

    compute_unit compute_unit_(
        .rst(rst),
        .clk(clk),
        .clr(contr_core1_clr),
        .ena(contr_core1_ena),
        .set_pc_req(contr_core1_set_pc_req),
        .set_pc_addr(contr_core1_set_pc_addr),

        .outflen(outflen),
        .out(out),
        .outen(outen),

        .halt(contr_core1_halt),

        .mem_addr(core1_mem_addr),
        .mem_rd_data(core1_mem_rd_data),
        .mem_wr_data(core1_mem_wr_data),
        .mem_ack(core1_mem_ack),
        .mem_busy(core1_mem_busy),
        .mem_rd_req(core1_mem_rd_req),
        .mem_wr_req(core1_mem_wr_req)
    );
endmodule
