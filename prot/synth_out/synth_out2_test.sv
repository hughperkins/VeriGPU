module use_synth_out(
    input clk,
    input rst,
    output reg out,
    input a
);
    reg child_out;
    // reg child_a;
    synth_out2 dut(
        .clk(clk),
        .rst(rst),
        .out(child_out),
        .a(a)
    );

    always @(*) begin
        out = child_out;
    end
endmodule

module synth_out_test2();
    reg clk;
    reg out;
    reg rst;
    reg a;

    synth_out2 synth_out_(
        .clk(clk),
        .out(out),
        .rst(rst),
        .a(a)
    );

    task pos();
        $display("  +");
        #5 clk = 1;
    endtask

    task neg();
        $display("-");
        #5 clk = 0;
    endtask

    task tick();
        $display("-");
        #5 clk = 0;
        $display("  +");
        #5 clk = 1;
    endtask

    initial begin
        rst = 0;
        a = 0;
        tick();
        tick();

        rst = 1;
        $monitor("a=%0b out=%0b", a, out);

        tick();
        assert(~out);

        tick();
        assert(~out);
        a = 1;

        neg();
        pos();

        neg();
        assert(out);
        pos();

        assert(out);

        tick();

        $display("done");

        tick();
        tick();

        $finish;
    end
endmodule
