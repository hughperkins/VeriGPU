module test_mem();
reg [15:0] read_addr, write_addr;
reg [7:0] read_data, write_data;
reg clk, we;

mem mem1(.read_addr(read_addr), .write_addr(write_addr), .clk(clk), .we(we),
    .read_data(read_data), .write_data(write_data));

initial begin
    clk = 1;
    forever begin
        #5 clk = ~clk;
    end
end

initial begin
    $monitor("t=%d read_addr=%d read_data=%d", $time(), read_addr, read_data);
    we = 1;
    write_addr = 16'd10;
    write_data = 8'd123;

    #10
    we = 0;
    write_data = 8'd111;
    read_addr = 16'd10;

    #10
    we = 0;
    write_data = 8'd111;
    read_addr = 16'd10;

    #10
    we = 1;
    write_data = 8'd111;
    write_addr = 16'd16;
    read_addr = 16'd10;

    #10
    we = 0;
    read_addr = 16'd10;

    #10
    we = 0;
    read_addr = 16'd16;

    #100 $finish();
end

endmodule
