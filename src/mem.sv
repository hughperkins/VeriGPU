// represents memory controller (plus memory that the memory is controlling)
module mem(
    input we, clk,
    input [15:0] read_addr,
    input [15:0] write_addr,
    output [15:0] read_data,
    input [15:0] write_data,

    input [15:0] oob_write_addr,
    input [15:0] oob_write_data,
    input oob_wen
);
    reg [15:0] mem [65536];

    always @(posedge clk) begin
        if (we) begin
            mem[write_addr] <= write_data;
        end
        if(oob_wen) begin
            mem[oob_write_addr] <= oob_write_data;
        end
    end
    assign read_data = mem[read_addr];
endmodule

/*
seems not supported by either iverilog, or verilator
interface mem_if;
    logic we, clk;
    wire [15:0] read_addr;
    wire [15:0] write_addr;
    logic [7:0] read_data;
    logic [7:0] write_data;
endinterface
*/
