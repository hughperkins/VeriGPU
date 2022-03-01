module comp_driver();
    reg rst, clk;
    reg [15:0] out;
    reg [15:0] mem [256];
    reg [7:0] op;
    reg [7:0] p1;
    comp comp1(.rst(rst), .clk(clk), .out(out), .mem(mem), .op(op), .p1(p1));
    initial begin
        clk = 1;
        forever #5 clk = ~clk;
    end
    initial begin
        $readmemh("build/prog3.hex", mem);
    end
    initial begin
        $monitor("t=%d rst=%b out=%d %h op=%h p1=%h", $time(), rst, out, out, op, p1);
        rst = 1;
        #10 rst = 0;
        #100 $finish();
    end
endmodule
