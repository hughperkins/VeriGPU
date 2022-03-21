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

        $display("a=%0f b=%0f out=%0f", to_real(_a), to_real(_b), to_real(out));
        `assert(reals_near(to_real(out), expected_out));
        #1;
    endtask

    initial begin
        test_add(1.0, 1.0, 2.0);
        test_add(1.23, 2.56, 3.79);
    end
endmodule
