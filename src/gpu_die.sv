// represents the contents of the GPU die, i.e. contains things like:
// GPU cores
// shared memory
// GPU controller
// global memory controller
//
// for now, the global memory controller also contains global memory, but we will split those
// off from each other

`default_nettype none
module gpu_die(
    input clk,
    input rst,

    // comms with mainboard cpu
    input [31:0] cpu_recv_instr,  
    // I'm using in/out, because less ambigous than rd/wr I feel, i.e. invariant
    // with PoV this module, or PoV calling module
    input [31:0] cpu_in_data,
    output reg [31:0] cpu_out_data,
    output reg cpu_out_ack,

    output reg halt,
    output reg outflen,
    output reg outen,
    output reg [data_width - 1:0] out
);
    wire core1_mem_rd_req;
    wire core1_mem_wr_req;

    wire [addr_width - 1:0] core1_mem_addr;
    wire [data_width - 1:0] core1_mem_rd_data;
    wire [data_width - 1:0] core1_mem_wr_data;

    wire core1_mem_busy;
    wire core1_mem_ack;

    wire contr_mem_wr_en;
    wire contr_mem_rd_en;
    wire [addr_width - 1:0] contr_mem_wr_addr;
    wire [data_width - 1:0] contr_mem_wr_data;
    wire [addr_width - 1:0] contr_mem_rd_addr;
    wire [data_width - 1:0] contr_mem_rd_data;
    wire contr_mem_rd_ack;

    wire contr_core1_ena;
    wire contr_core1_clr;
    wire contr_core1_set_pc_req;
    wire [data_width - 1:0] contr_core1_set_pc_addr;
    wire contr_core1_halt;

    reg core1_halt;

    global_mem_controller global_mem_controller_(
        .clk(clk),
        .rst(rst),
        // .ena(ena),

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

        // .oob_wr_addr(oob_wr_addr), .oob_wr_data(oob_wr_data),
        // .oob_wen(oob_wen)

        // .contr_wr_addr(oob_wr_addr), .contr_wr_data(oob_wr_data),
        // .contr_wen(oob_wen)
    );

    core core1(
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

    gpu_controller gpu_controller_(
        .rst(rst),
        .clk(clk),

        .cpu_recv_instr(cpu_recv_instr),
        .cpu_in_data(cpu_in_data),
        .cpu_out_data(cpu_out_data),
        .cpu_out_ack(cpu_out_ack),

        .mem_wr_en(contr_mem_wr_en),
        .mem_rd_en(contr_mem_rd_en),
        .mem_wr_addr(contr_mem_wr_addr),
        .mem_wr_data(contr_mem_wr_data),
        .mem_rd_addr(contr_mem_rd_addr),
        .mem_rd_data(contr_mem_rd_data),
        .mem_rd_ack(contr_mem_rd_ack),

        .core_ena(contr_core1_ena),
        .core_clr(contr_core1_clr),
        .core_halt(contr_core1_halt),
        .core_set_pc_req(contr_core1_set_pc_req),
        .core_set_pc_addr(contr_core1_set_pc_addr)
    );
endmodule
