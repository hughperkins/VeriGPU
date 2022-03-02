// test having a bus between modules

module comp(
    input [7:0] read_data,
    output reg [7:0] write_data,
    output reg [7:0] read_addr,
    input clk,
    input rst,
    output reg [7:0] out
);
    reg [2:0] cnt;
    always @(posedge clk, posedge rst) begin
        if (rst) begin
            cnt <= 0;
        end else begin
            read_addr = cnt;
            out <= read_data;
            cnt <= cnt + 1;
        end
    end
endmodule

module mem(
    output reg [7:0] read_data,
    input [7:0] write_data,
    input [7:0] read_addr,
    input clk,
    input [7:0] startup_write_data,
    input [7:0] startup_write_addr,
    input startup_wen
);
    reg [7:0] mem[256];
    initial begin
        $display("hello");
        $monitor("mem read_addr=%d read_data=%d clk=%d swen=%b", read_addr, read_data, clk, startup_wen);
        #200
        $finish;
    end
    always @(posedge clk) begin
        // assign read_data = 8'd123;
        // read_data = 8'd123;
        if (startup_wen) begin
            mem[startup_write_addr] <= startup_write_data;
        end else begin
            read_data <= mem[read_addr];
        end
    end
endmodule

module test_comp();
    wire [7:0] read_data;
    wire [7:0] write_data;
    reg clk;
    reg startup_wen;
    reg [7:0] startup_write_data;
    reg [7:0] startup_write_addr;
    wire [7:0] out;
    wire [7:0] read_addr;
    reg rst;

    comp comp1(
        .read_data(read_data), .write_data(write_data), .clk(clk), .out(out), .rst(rst),
        .read_addr(read_addr)
    );
    mem mem1(
        .read_data(read_data), .write_data(write_data), .clk(clk),
        .read_addr(read_addr),
        .startup_write_data(startup_write_data),
        .startup_write_addr(startup_write_addr),
        .startup_wen(startup_wen)
    );

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%d out=%d, read_addr=%d read_data=%d", $time, out, read_addr, read_data);
        rst = 1;
        startup_wen = 1;
        for(int i = 0; i < 8; i++) begin
            #10
            startup_write_data = i + 3;
            startup_write_addr = i;
        end
        #10
        startup_wen = 0;
        #10 rst = 0;
        #100
        $finish();
    end

endmodule
