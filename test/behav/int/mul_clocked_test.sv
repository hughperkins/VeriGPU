parameter width = 32;

module mul_clocked_test();
    reg                     clk;
    reg                     rst;
    reg                     req;
    reg                     ack;
    reg [width - 1:0] a;
    reg [width - 1:0] b;
    reg [width - 1:0] out;

    mul_pipeline_32bit mul_pipeline_32bit_(
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

    task test_mul(input [width - 1:0] _a, input [width - 1:0] _b, input [width - 1:0] expected_out);
        int cnt;

        `assert(~ack);
        $display("submitting req %0d * %0d", _a, _b);
        a <= _a;
        b <= _b;
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

        `assert(ack);
        $display("test_mul a=%0d b=%0d out=%0d cnt=%0d", _a, _b, out, cnt);
        `assert(out == expected_out);

        tick();
        `assert(~ack);
    endtask

    initial begin
        // $monitor(
        //     "t=%0d test.mon a=%0d b=%0d out=%0d ack=%b",
        //     $time, a, b, out, ack);
        rst = 1;
        req = 0;
        clk = 0;

        tick();
        $display("reset going off");
        rst <= 0;

        tick();

        test_mul(3, 5, 15);
        test_mul(15, 4, 60);
        test_mul(15, 0, 0);
        test_mul(3'b111, 2'b11, 21);  // 7 * 3 = 21
        test_mul(7'b1111111, 5'b11111, 7'b1111111 * 5'b11111);  // 127 * 31 = 3937
        test_mul(150, 4, 600);
        test_mul(150, 40, 6000);
        test_mul(1254424, 124, 1254424 * 124);
        test_mul(9556, 124, 9556 * 124);
        test_mul(95562, 124, 95562 * 124);
        test_mul(347911, 12345, 4294961295);
    end
endmodule
