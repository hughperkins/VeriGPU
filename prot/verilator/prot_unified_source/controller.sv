/*
This will manage communicaitons with the cpu/driver.

Will handle things like:
- allocating gpu memory
- copy data to/from gpu
- copy kernel to gpu
- launching kernels
*/
module controller(
    input clk,
    input rst,
    input instr,
    input reg rd_req,
    input [31:0] rd_addr,
    input [31:0] rd_data,
    input reg wr_req,
    input [31:0] wr_addr,
    output reg [31:0] wr_data
);

endmodule;
