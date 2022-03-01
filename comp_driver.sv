module comp_driver();
    reg rst, clk;
    reg [15:0] out;
    reg [15:0] mem [64];
    comp comp1(.rst(rst), .clk(clk), .out(out), .mem(mem));
    initial begin
        clk = 1;
        forever #5 clk = ~clk;
    end
    initial begin
        $readmemh("build/prog2.hex", mem);
    end
    initial begin
        $monitor("t=%d rst=%b out=%d %h", $time(), rst, out, out);
        rst = 1;
        #10 rst = 0;
        #100 $finish();
    end
endmodule
