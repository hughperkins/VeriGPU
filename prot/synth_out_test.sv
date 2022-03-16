module synth_out_test();
    reg clk;
    reg out;
    reg rst;
    reg a;

    synth_out synth_out1(
        .clk(clk),
        .out(out),
        .rst(rst),
        .a(a)
    );

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("clk=%0b rst=%0b out=%0b a=%0b", clk, rst, out, a);
        rst = 1;
        #5
        a = 0;
        #10
        rst = 0;
        #10

        assert(~out);

        #10
        assert(~out);
        a = 1;

        #10;
        #0;
        assert(out);

        #9
        assert(out);

        #10;
        $display("done");

        #100 $finish;
    end
endmodule
