// test having a bus between modules

module comp(
    input [7:0] read_data,
    output reg [7:0] write_data,
    output reg [7:0] write_addr,
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
            out <= read_data;
            cnt <= cnt + 1;
            write_addr <= cnt + 10;
            write_data <= cnt + 5;
        end
    end
    assign read_addr = cnt;
endmodule

module mem(
    output reg [7:0] read_data,
    input [7:0] write_data,
    input [7:0] write_addr,
    input [7:0] read_addr,
    input clk,
    input [7:0] dbg_write_data,
    input [7:0] dbg_write_addr,
    input [7:0] dbg_read_addr,
    output reg [7:0] dbg_read_data,
    input dbg_wen
);
    reg [7:0] mem[256];
    /*
    initial begin
        $display("hello");
        $monitor("mem read_addr=%d read_data=%d clk=%d swen=%b", read_addr, read_data, clk, dbg_wen);
        #200
        $finish;
    end
    */
    always @(posedge clk) begin
        // assign read_data = 8'd123;
        // read_data = 8'd123;
        if (dbg_wen) begin
            mem[dbg_write_addr] <= dbg_write_data;
        end
        else begin
            mem[write_addr] <= write_data;
        end
    end
    assign read_data = mem[read_addr];
    assign dbg_read_data = mem[dbg_read_addr];
endmodule

module test_comp();
    wire [7:0] read_data;
    wire [7:0] write_data;
    wire [7:0] write_addr;
    reg clk;
    reg dbg_wen;
    reg [7:0] dbg_write_data;
    reg [7:0] dbg_write_addr;
    reg [7:0] dbg_read_data;
    reg [7:0] dbg_read_addr;
    wire [7:0] out;
    wire [7:0] read_addr;
    reg rst;

    comp comp1(
        .read_data(read_data), .write_data(write_data), .clk(clk), .out(out), .rst(rst),
        .read_addr(read_addr), .write_addr(write_addr)
    );
    mem mem1(
        .read_data(read_data), .write_data(write_data), .clk(clk),
        .read_addr(read_addr), .write_addr(write_addr),
        .dbg_write_data(dbg_write_data),
        .dbg_write_addr(dbg_write_addr),
        .dbg_read_addr(dbg_read_addr), .dbg_read_data(dbg_read_data),
        .dbg_wen(dbg_wen)
    );

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%d out=%d, read_addr=%d read_data=%d dbg_read_addr=%d dbg_read_data=%d", $time, out, read_addr, read_data, dbg_read_addr, dbg_read_data);
        rst = 1;
        dbg_wen = 1;
        for(int i = 0; i < 8; i++) begin
            #10
            dbg_write_data = i + 3;
            dbg_write_addr = i;
        end
        #10
        dbg_wen = 0;
        #10 rst = 0;
        #100 rst = 1;
        for(int i = 0; i < 8; i++) begin
            #10
            dbg_read_addr = i + 10;
        end
        $finish();
    end

endmodule
