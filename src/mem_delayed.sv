// represents memory controller (plus memory that the memory is controlling)

// forked from mem.sv. Introduces some delays, to simulate that memory
// requests in reality typically take a while...
module mem(
    input we, clk,
    input [15:0] rd_addr,
    input [15:0] wr_addr,
    output [15:0] rd_data,
    input [15:0] wr_data,
    output reg rd_ready;

    input [15:0] oob_wr_addr,
    input [15:0] oob_wr_data,
    input oob_wen
);
    reg [15:0] mem [64];

    parameter delay_cycles = 5;
    reg [15:0] rd_addr_delayed [delay_cycles];
    reg [15:0] wr_addr_delayed [delay_cycles];

    always @(posedge clk) begin
        if (we) begin
            mem[wr_addr_delayed[delay_cycles - 1]] <= wr_data;
        end
        if(oob_wen) begin
            mem[oob_wr_addr] <= oob_wr_data;
        end
        rd_addr_delayed[0] <= rd_addr;
        wr_addr_delayed[0] <= wr_addr;
        for(int i = 1; i < 5; i++) begin
           read_addr_delayed[i] <= read_addr_delayed[i - 1];
           write_addr_delayed[i] <= write_addr_delayed[i - 1];
        end
    end
    // assign read_data = mem[read_addr];
    assign read_data = mem[read_addr_delayed[delay_cycles - 1]];
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
