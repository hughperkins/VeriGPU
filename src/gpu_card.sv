// represents an entire GPU card, including both the GPU die, and the GPU global memory
// (although for now, the global_mem_controller contains the global memory)
`default_nettype none
module gpu_card(
    input clk,
    input rst,
    output halt,

    output outflen,
    output outen,
    output [data_width - 1:0] out,

    input [31:0] cpu_recv_instr,  
    input [31:0] cpu_in_data,
    output reg [31:0] cpu_out_data,
    output reg cpu_out_ack
);
    gpu_die gpu_die_(
        .clk(clk),
        .rst(rst),
        .halt(halt),
        .cpu_recv_instr(cpu_recv_instr),
        .cpu_in_data(cpu_in_data),
        .cpu_out_data(cpu_out_data),
        .cpu_out_ack(cpu_out_ack),

        .outflen(outflen),
        .outen(outen),
        .out(out)
    );

    // global_mem will go here
    // global_mem global_mem_();
endmodule
