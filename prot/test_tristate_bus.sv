module mem(
    input [7:0] addr,
    inout [7:0] data,
    input clk,
    input write
);
    reg [7:0] mem[65536];
    always @(posedge clk) begin
        if (write) begin
            mem[addr] <= data;
        end
    end
    assign data = write ? 8'bz : mem[addr];
endmodule

module comp(
    output reg [7:0] addr,
    inout [7:0] data,
    input clk,
    input rst,
    output reg write,
    output reg [7:0] out
);
    reg [7:0] write_value;
    reg [3:0] cnt;

    always @(posedge clk, negedge rst) begin
        if (~rst) begin
            cnt <= '0;
        end else begin
            case(cnt)
                0: begin
                    write <= 1;
                    write_value <= 8'd123;
                    addr <= 8'd1;
                end
                1: begin
                    write <= 1;
                    write_value <= 8'd111;
                    addr <= 8'd2;
                end
                2: begin
                    write <= 1;
                    write_value <= 8'd222;
                    addr <= 8'd3;
                end
                3: begin
                    write <= 0;
                    addr <= 8'd1;
                    out <= data;
                end
                4: begin
                    write <= 0;
                    addr <= 8'd2;
                    out <= data;
                end
                5: begin
                    write <= 0;
                    addr <= 8'd3;
                    out <= data;
                end
            endcase
            cnt <= cnt + 1;
        end
    end

    assign data = write ? write_value : 8'bz;
    assign value = write ? 8'b0 : data;
endmodule

module test_driver(
);
    reg clk, write;
    reg [7:0] addr;
    wire [7:0] data;
    reg [7:0] out;
    reg rst;

    mem mem_(.addr(addr), .data(data), .clk(clk), .write(write) );
    comp comp_(.addr(addr), .data(data), .clk(clk), .write(write), .out(out), .rst(rst) );

    initial begin
        clk = 1;
        forever
            #5 clk = ~clk;
    end

    initial begin
        $monitor("t=%d write=%b addr=%d data=%d out=%d", $time, write, addr, data, out);
        rst = 0;
        # 10 rst = 1;
        # 100 $finish();
    end
endmodule
