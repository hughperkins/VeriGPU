/*
Targeting testing using verilator, whihc will provide a clock, and reset

The idea is that this module does most fo the work; so can easily be run by a lightweight
head from whatever simulator, eg pure verilog driver, or verilator c++ driver.
*/

module float_mul_pipeline_test_vtor(
    input clk,
    input rst,
    output reg finish,
    output reg fail,
    output reg [31:0] cnt,
    output reg [31:0] test_num,
    output reg [float_width - 1:0] out,
    output reg                     ack
);
    reg                     req;
    reg [float_width - 1:0] a;
    reg [float_width - 1:0] b;
    reg submitted;

    float_mul_pipeline float_mul_pipeline_(
        .clk(clk),
        .rst(rst),
        .req(req),
        .ack(ack),
        .a(a),
        .b(b),
        .out(out)
    );

    task test_mul(real _a, real _b, real _expected);
        if(~submitted) begin
            a <= make_float(_a);
            b <= make_float(_b);
            req <= 1;
            submitted <= 1;
        end else begin
            req <= 0;
            cnt <= cnt + 1;
            if(ack) begin
                $display("out=%0f", to_real(out));
                assert(reals_near(to_real(out), _expected));
                test_num <= test_num + 1;
                submitted <= 0;
                cnt <= 0;
            end
            assert(cnt < 100);
        end
    endtask;

    always @(posedge clk, posedge rst) begin
        if(rst) begin
            finish <= 0;
            fail <= 0;
            cnt <= 0;
            test_num <= 0;
            submitted <= 0;

            req <= 0;
            ack <= 0;
            a <= 0;
            b <= 0;
            out <= 0;
        end else begin
            if(~finish) begin
                case(test_num)
                    0: test_mul(1.2, 3.5, 4.2);
                    1: test_mul(100.0, 4.5, 450.0);

                    2: test_mul(0.0, 1.0, 0.0);
                    3: test_mul(1.0, 0.0, 0.0);
                    4: test_mul(0.0, 0.0, 0.0);

                    5: test_mul(1.0, 1.0, 1.0);
                    6: test_mul(1.1, 1.1, 1.21);
                    7: test_mul(11.0, 11.0, 121.0);
                    8: test_mul(1.9, 1.9, 3.61);

                    9: test_mul(1.0, 2.0, 2.0);
                    10: test_mul(2.0, 1.0, 2.0);
                    11: test_mul(2.0, 2.0, 4.0);
                    12: test_mul(2.0, 2.3, 4.6);
                    13: test_mul(8.0, 4.0, 32.0);
                    14: test_mul(10.0, 4.0, 40.0);
                    15: test_mul(10.1, 4.0, 40.4);
                    16: test_mul(101.0, 4.0, 404.0);
                    17: test_mul(100.0, 4.5, 450.0);
                    18: test_mul(20.0, 2.3, 46.0);
                    19: test_mul(200.0, 2.3, 460.0);
                    20: test_mul(200.0, 100.0, 20000.0);
                    21: test_mul(2000.0, 2.3, 4600.0);

                    22: test_mul(-2000.0, 2.3, -4600.0);
                    23: test_mul(2000.0, -2.3, -4600.0);
                    24: test_mul(-2000.0, -2.3, 4600.0);

                    25: finish <= 1;
                endcase
                // cnt <= cnt + 1;
            end
        end
    end
endmodule
