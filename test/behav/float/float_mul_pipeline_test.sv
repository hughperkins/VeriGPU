module float_mul_pipeline_test();
    reg                     clk;
    reg                     rst;
    reg                     req;
    reg                     ack;
    reg [float_width - 1:0] a;
    reg [float_width - 1:0] b;
    reg [float_width - 1:0] out;

    float_mul_pipeline float_mul_pipeline_(
        .clk(clk),
        .rst(rst),
        .req(req),
        .ack(ack),
        .a(a),
        .b(b),
        .out(out)
    );

    task pos();
        // $display("  +");
        #5 clk = 1;
    endtask

    task neg();
        // $display("-");
        #5 clk = 0;
    endtask

    task tick();
        // $display("-");
        #5 clk = 0;
        // $display("  +");
        #5 clk = 1;
    endtask

    task test_mul_zero(input real _a, input real _b, input real expected_out);
        // fewer cycles than test_mul
        `assert(~ack);
        $display("submitting zero req %f + %f", _a, _b);
        a <= make_float(_a);
        b <= make_float(_b);
        req <= 1;

        tick();
        req <= 0;
        `assert(~ack);

        tick();
        `assert(ack);
        $display("test_mul a=%0f b=%0f out=%0f", _a, _b, to_real(out));
        `assert(reals_near(to_real(out), expected_out));

        tick();
        `assert(~ack);
    endtask

    task test_mul(input real _a, input real _b, input real expected_out);
        int cnt;

        `assert(~ack);
        $display("submitting req %f * %f", _a, _b);
        a <= make_float(_a);
        b <= make_float(_b);
        req <= 1;

        tick();
        req <= 0;
        `assert(~ack);

        cnt = 0;
        do begin
            tick();
            // $display("out %0d", out);
            cnt = cnt + 1;
        end while(~ack && cnt < 80);
        // `assert(~ack);

        // tick();
        // $display("out %0d", out);
        // `assert(~ack);

        // tick();
        // $display("out %0d", out);
        // `assert(~ack);

        // tick();
        // $display("out %0d", out);
        // `assert(~ack);

        // tick();
        $display("cnt=%0d ack=%b", cnt, ack);
        `assert(ack);
        $display("test_mul a=%0f b=%0f out=%0f", _a, _b, to_real(out));
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
        rst = 1;
        req = 0;
        clk = 0;

        tick();
        $display("reset going off");
        rst <= 0;

        tick();

        test_mul_zero(0.0, 1.0, 0.0);
        test_mul_zero(1.0, 0.0, 0.0);
        test_mul_zero(0.0, 0.0, 0.0);

        test_mul(1.0, 1.0, 1.0);
        test_mul(1.1, 1.1, 1.21);
        test_mul(11.0, 11.0, 121.0);
        test_mul(1.9, 1.9, 3.61);

        test_mul(1.0, 2.0, 2.0);
        test_mul(2.0, 1.0, 2.0);
        test_mul(2.0, 2.0, 4.0);
        test_mul(2.0, 2.3, 4.6);
        test_mul(8.0, 4.0, 32.0);
        test_mul(10.0, 4.0, 40.0);
        test_mul(10.1, 4.0, 40.4);
        test_mul(101.0, 4.0, 404.0);
        test_mul(100.0, 4.5, 450.0);
        test_mul(20.0, 2.3, 46.0);
        test_mul(200.0, 2.3, 460.0);
        test_mul(200.0, 100.0, 20000.0);
        test_mul(2000.0, 2.3, 4600.0);

        test_mul(-2000.0, 2.3, -4600.0);
        test_mul(2000.0, -2.3, -4600.0);
        test_mul(-2000.0, -2.3, 4600.0);
    end
endmodule
