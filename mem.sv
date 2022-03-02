module mem(
    input we, clk,
    input [15:0] read_addr,
    input [15:0] write_addr,
    output [7:0] read_data,
    input [7:0] write_data
);
    reg [7:0] mem [65536];

    always @(posedge clk) begin
        if (we) begin
            mem[write_addr] <= write_data;
        end
    end
    assign read_data = mem[read_addr];
endmodule
