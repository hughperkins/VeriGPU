module dadda_24bit_test();
    parameter width = 24;

    reg [width - 1:0] a;
    reg [width - 1:0] b;
    reg [width * 2 - 1:0] out;

    dadda_24bit48 dadda_(
        .a(a),
        .b(b),
        .out(out)
    );

    task test_mul(input [width - 1:0] a_, input [width - 1:0] b_, input [width * 2 - 1:0] expected_out);
        a = a_;
        b = b_;
        #1;
        $display("a %0d b %0d out %0d %0d", a, b, out, expected_out);
        `assert(out == expected_out);
        #1;
    endtask

    initial begin
        test_mul(3, 5, 15);
        test_mul(15, 4, 60);
        test_mul(15, 0, 0);
        test_mul(9556, 124, 9556 * 124);
        test_mul(95562, 124, 95562 * 124);
        test_mul( {24{1'b1}}, {24{1'b1}}, 281474943156225);
    end
endmodule
