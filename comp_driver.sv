module comp_driver();
    reg rst, clk;
    reg [15:0] out;
    reg [15:0] mem [256];
    reg [3:0] op;
    reg [3:0] reg_select;
    reg [7:0] p1;
    //reg [7:0] regs[16];
    reg [7:0] x1;

    comp comp1(
        .rst(rst), .clk(clk), .out(out), .mem(mem), .op(op), .p1(p1),
        .reg_select(reg_select),
        .x1(x1)
    );
    initial begin
        clk = 1;
        forever #5 clk = ~clk;
    end
    initial begin
        $readmemh("build/{PROG}.hex", mem);
    end
    initial begin
        $monitor(
            "t=%d rst=%b out=%h op=%h p1=%h rs=%h x1=%h",
            $time(), rst, out,  op,   p1,   reg_select, x1);
        rst = 1;
        #10 rst = 0;
        #200 $finish();
    end
endmodule
