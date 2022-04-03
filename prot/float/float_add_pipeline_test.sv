module float_add_test();
    reg                     clk;
    reg                     rst;
    reg                     req;
    reg                     ack;
    reg [float_width - 1:0] a;
    reg [float_width - 1:0] b;
    reg [float_width - 1:0] out;

    float_add_pipeline float_add_pipeline_(
        .clk(clk),
        .rst(rst),
        .req(req),
        .ack(ack),
        .a(a),
        .b(b),
        .out(out)
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

    task test_add(input real _a, input real _b, input real expected_out);
        `assert(~ack);
        $display("submitting req %f + %f", _a, _b);
        a <= make_float(_a);
        b <= make_float(_b);
        req <= 1;

        tick();
        req <= 0;
        `assert(~ack);

        tick();
        $display("out %0d", out);
        `assert(~ack);

        tick();
        `assert(~ack);

        tick();
        `assert(ack);
        $display("test_add a=%0f b=%0f out=%0f", _a, _b, to_real(out));
        `assert(reals_near(to_real(out), expected_out));

        tick();
        `assert(~ack);
    endtask

    real a_real, b_real, out_real;
    assign a_real = to_real(a);
    assign b_real = to_real(b);
    assign out_real = to_real(out);

    initial begin
        $monitor(
            "t=%0d test.mon a=%0f b=%0f out=%0f",
            $time, a_real, b_real, out_real);
        rst <= 0;
        req <= 0;
        clk <= 0;

        tick();
        $display("reset going off");
        rst <= 1;

        tick();

        test_add(1.0, 1.0, 2.0);
        test_add(1.23, 2.56, 3.79);
        test_add(1.23, 0.00456, 1.23456);
        test_add(0.000123, 0.000000456, 0.000123456);
        test_add(2.0, 3.0, 5.0);
        test_add(2000.0, 300.0, 2300.0);

        // both neg
        test_add(-5.1, -3.2, -8.3);

        // one neg
        test_add(1.5, -1.25, 0.25);
        test_add(1.25, -1.5, -0.25);
        test_add(-1.5, 1.25, -0.25);
        test_add(-1.25, 1.5, 0.25);

        // neg to zero
        test_add(1.5, -1.5, 0.0);
    end
endmodule
