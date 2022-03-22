module float_mul_test();
    reg [float_width - 1:0] a;
    reg [float_width - 1:0] b;
    reg [float_width -1:0] out;

    float_mul float_mul_(
        .a(a),
        .b(b),
        .out(out)
    );

    task test_mul(input real _a, input real _b, input real expected_out);
        a <= make_float(_a);
        b <= make_float(_b);
        #1

        $display("test_add a=%0f b=%0f out=%0f", _a, _b, to_real(out));
        `assert(reals_near(to_real(out), expected_out));
        #1;
    endtask

    initial begin
        test_mul(0.0, 1.0, 0.0);
        test_mul(1.0, 0.0, 0.0);
        test_mul(0.0, 0.0, 0.0);

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
