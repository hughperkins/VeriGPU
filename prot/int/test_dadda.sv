module test_dadda();
    reg [data_width - 1:0] a;
    reg [data_width - 1:0] b;
    reg [data_width - 1:0] out;

    dadda_32bit dadda_(
        .a(a),
        .b(b),
        .out(out)
    );

    task test_mul(input [data_width - 1:0] a_, input [data_width - 1:0] b_, input [data_width - 1:0] expected_out);
        a = a_;
        b = b_;
        #1;
        $display("a %0d b %0d out %0d", a, b, out);
        `assert(out == expected_out);
        #1;
    endtask

    initial begin
        test_mul(3, 5, 15);
        test_mul(15, 4, 60);
        test_mul(15, 0, 0);
        test_mul(1254424, 124, 1254424 * 124);
    end
endmodule;
