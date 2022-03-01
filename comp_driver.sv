module comp_driver();
    reg rst, clk;
    reg [15:0] out;
    reg [15:0] mem [32];
    comp comp1(.rst(rst), .clk(clk), .out(out), .mem(mem));
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    initial begin
        $readmemh("prog1.hex", mem);
    end
    initial begin
        $monitor("t=%d clk=%d rst=%b out=%d %h", $time(), clk, rst, out, out);
        rst = 1;
        #10 rst = 0;
        #100 $finish();
    end
endmodule
