// represents an entire GPU card, including both the GPU die, and the GPU global memory
// (although for now, the global_mem_controller contains the global memory)
module gpu_card(
    input clk,
    input rst,
    output halt,

    output outflen,
    output outen,
    output [data_width - 1:0] out,

    input [31:0] cpu_recv_instr,  
    input [31:0] cpu_in_data,
    output reg [31:0] cpu_out_data
);
    gpu_die gpu_die_(
        .clk(clk),
        .rst(rst),
        .halt(halt),
        .cpu_recv_instr(cpu_recv_instr),
        .cpu_in_data(cpu_in_data),
        .cpu_out_data(cpu_out_data),

        .outflen(outflen),
        .outen(outen),
        .out(out)
    );

    // global_mem will go here
    // global_mem global_mem_();
endmodule
