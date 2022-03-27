parameter width = 32;

module mul_test();
    reg [width - 1:0] a;
    reg [width - 1:0] b;
    reg [width - 1:0] out;

    mul mul_(
        .a(a),
        .b(b),
        .out(out)
    );

    task test_mul(input [width - 1:0] a_, input [width - 1:0] b_, input [width - 1:0] expected_out);
        a = a_;
        b = b_;
        #1;
        $display("a %0d b %0d out %0d %b", a, b, out, out);
        `assert(out[31:0] == expected_out);
        #1;
    endtask

    initial begin
        test_mul(1, 1, 1);
        test_mul(1, 2, 2);
        test_mul(2, 1, 2);
        test_mul(1, 3, 3);
        test_mul(3, 1, 3);
        test_mul(2, 3, 6);
        test_mul(3, 3, 9);

        test_mul(4, 2, 8);
        test_mul(4, 3, 12);
        test_mul(5, 3, 15);

        test_mul(3, 5, 15);
        test_mul(3, 5, 15);
        test_mul(15, 4, 60);
        test_mul(15, 0, 0);
        test_mul(2'b11, 2'b11, 9);
        test_mul(31, 3, 93);
        test_mul(31, 6, 186);
        test_mul(31, 8, 248);

        test_mul(257, 255, 65535);
        test_mul(255, 257, 65535);

        test_mul(9556, 124, 9556 * 124);  // 1184944
        test_mul(95562, 124, 95562 * 124);
        test_mul(1254424, 124, 1254424 * 124);  // 155548576
        test_mul(347911, 12345, 4294961295);
    end
endmodule
