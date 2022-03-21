module float_add_test();
    reg [float_width - 1:0] a;
    reg [float_width - 1:0] b;
    reg [float_width -1:0] out;

    float_add float_add_(
        .a(a),
        .b(b),
        .out(out)
    );

    task test_add(input real _a, input real _b, input real expected_out);
        a <= make_float(_a);
        b <= make_float(_b);
        #1

        $display("test_add a=%0f b=%0f out=%0f", _a, _b, to_real(out));
        `assert(reals_near(to_real(out), expected_out));
        #1;
    endtask

    initial begin
        test_add(1.0, 1.0, 2.0);
        test_add(1.23, 2.56, 3.79);
        test_add(1.23, 0.00456, 1.23456);
        test_add(0.000123, 0.000000456, 0.000123456);
        test_add(2.0, 3.0, 5.0);
        test_add(2000.0, 300.0, 2300.0);
    end
endmodule
