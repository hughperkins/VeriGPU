/*

two read ports for the processor

one write port for the processor
one write port for the apu

*/

module reg_file(
    input clk, rst,

    input [reg_sel_width - 1:0] proc_rs1_sel,
    input [reg_sel_width - 1:0] proc_rs2_sel,
    output reg [data_width - 1:0] proc_rs1_data,
    output reg [data_width - 1:0] proc_rs2_data,

    input proc_wr_req,
    output reg proc_ack,
    input [reg_sel_width - 1:0] proc_rd_sel,
    input [data_width - 1:0] proc_rd_data,

    input apu_wr_req,
    output reg apu_ack,
    input [reg_sel_width - 1:0] apu_wr_sel,
    input [data_width - 1:0] apu_wr_data
);
    reg [data_width - 1:0] regs [num_regs];

    always @(posedge clk, posedge rst) begin
        if(rst) begin
            proc_ack <= 0;
            apu_ack <= 0;
        end else begin
            proc_ack <= 0;
            apu_ack <= 0;
            if(proc_wr_req) begin
                regs[proc_rd_sel] <= proc_rd_data;
                proc_ack <= 1;
            end
            if(apu_wr_req) begin
                regs[apu_wr_sel] <= apu_wr_data;
                apu_ack <= 1;
            end
        end
    end

    assign proc_rs1_data = regs[proc_rs1_sel];
    assign proc_rs2_data = regs[proc_rs2_sel];
endmodule
