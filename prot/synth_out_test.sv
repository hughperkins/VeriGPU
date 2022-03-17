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
        // posedge
        rst = 1;

        #10; // posedge
        a = 0;

        #10; // posedge
        rst = 0;

        #5;// negedge
        assert(~out);

        #5; // posedge

        #5;// negedge
        assert(~out);

        #5; // posedge
        a = 1;

        #10; // posedge

        #5; // negedge
        assert(out);

        #5; // posedge

        #5; // negedge
        assert(out);

        #5;
        $display("done");

        #100 $finish;
    end
endmodule
