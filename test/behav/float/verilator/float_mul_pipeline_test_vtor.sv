/*
Targeting testing using verilator, whihc will provide a clock
*/

module float_mul_pipeline_test_vtor(
    input clk,
    input rst,
    output reg finish,
    output reg fail,
    output reg [31:0] cnt,
    output reg [31:0] test_num,
    output reg [float_width - 1:0] out
);
    reg                     req;
    reg                     ack;
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

    //     test_mul_zero(0.0, 1.0, 0.0);
    //     test_mul_zero(1.0, 0.0, 0.0);
    //     test_mul_zero(0.0, 0.0, 0.0);

    //     test_mul(1.0, 1.0, 1.0);
    //     test_mul(1.1, 1.1, 1.21);
    //     test_mul(11.0, 11.0, 121.0);
    //     test_mul(1.9, 1.9, 3.61);

    //     test_mul(1.0, 2.0, 2.0);
    //     test_mul(2.0, 1.0, 2.0);
    //     test_mul(2.0, 2.0, 4.0);
    //     test_mul(2.0, 2.3, 4.6);
    //     test_mul(8.0, 4.0, 32.0);
    //     test_mul(10.0, 4.0, 40.0);
    //     test_mul(10.1, 4.0, 40.4);
    //     test_mul(101.0, 4.0, 404.0);
    //     test_mul(100.0, 4.5, 450.0);
    //     test_mul(20.0, 2.3, 46.0);
    //     test_mul(200.0, 2.3, 460.0);
    //     test_mul(200.0, 100.0, 20000.0);
    //     test_mul(2000.0, 2.3, 4600.0);

    //     test_mul(-2000.0, 2.3, -4600.0);
    //     test_mul(2000.0, -2.3, -4600.0);
    //     test_mul(-2000.0, -2.3, 4600.0);

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
                    0: begin
                        if(~submitted) begin
                            a <= make_float(1.2);
                            b <= make_float(3.5);
                            req <= 1;
                            submitted <= 1;
                        end else begin
                            req <= 0;
                            if(ack) begin
                                $display("out=%0f", to_real(out));
                                assert(reals_near(to_real(out), 4.2));
                                // finish <= 1;
                                test_num <= test_num + 1;
                                submitted <= 0;
                            end
                        end
                    end
                    1: begin
                        // 100.0, 4.5, 450.0
                        if(~submitted) begin
                            a <= make_float(100.0);
                            b <= make_float(4.5);
                            req <= 1;
                            submitted <= 1;
                        end else begin
                            req <= 0;
                            if(ack) begin
                                $display("out=%0f", to_real(out));
                                assert(reals_near(to_real(out), 450.0));
                                finish <= 1;
                                // test_num += 1;
                                submitted <= 0;
                            end
                        end
                    end
                endcase
                cnt <= cnt + 1;
            end
        end
    end
endmodule
