/*

two read ports for the processor

one write port for the processor
one write port for the apu

*/

module reg_file(
    input clk, rst,

    input [reg_sel_width - 1:0] rs1_sel,
    input [reg_sel_width - 1:0] rs2_sel,
    output reg [data_width - 1:0] rs1_data,
    output reg [data_width - 1:0] rs2_data,

    input wr_req,
    // output reg wr_ack,
    input [reg_sel_width - 1:0] wr_sel,
    input [data_width - 1:0] wr_data
);
    reg [data_width - 1:0] regs [num_regs];

    always @(*) begin
        // proc_ack = 0;

        rs1_data = regs[rs1_sel];
        rs2_data = regs[rs2_sel];

        // if(rd_req) begin
        //     regs[rd_sel] <= proc_rd_data;
        //     proc_ack <= 1;
        // end
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            regs[0] <= '0;
        end else begin
            if(wr_req && (wr_sel != 0)) begin
                regs[wr_sel] <= wr_data;
                // proc_ack <= 1;
            end
        end
    end

    // assign proc_rs1_data = regs[proc_rs1_sel];
    // assign proc_rs2_data = regs[proc_rs2_sel];
endmodule
