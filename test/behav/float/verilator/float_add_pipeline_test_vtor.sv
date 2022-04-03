/*
Targeting testing using verilator, whihc will provide a clock, and reset

The idea is that this module does most fo the work; so can easily be run by a lightweight
head from whatever siadd ator, eg pure verilog driver, or verilator c++ driver.
*/

module float_add_pipeline_test_vtor(
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

    float_add_pipeline float_add_pipeline_(
        .clk(clk),
        .rst(rst),
        .req(req),
        .ack(ack),
        .a(a),
        .b(b),
        .out(out)
    );

    task test_add (real _a, real _b, real _expected);
        if(~submitted) begin
            a <= make_float(_a);
            b <= make_float(_b);
            req <= 1;
            submitted <= 1;
        end else begin
            req <= 0;
            cnt <= cnt + 1;
            if(ack) begin
                $display("a=%0f b=%0f out=%0f expected=%0f", _a, _b, to_real(out), _expected);
                assert(reals_near(to_real(out), _expected));
                test_num <= test_num + 1;
                submitted <= 0;
                cnt <= 0;
            end
            assert(cnt < 100);
        end
    endtask;

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
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
                    0: test_add(1.0, 1.0, 2.0);
                    1: test_add(1.23, 2.56, 3.79);
                    2: test_add(1.23, 0.00456, 1.23456);
                    3: test_add(0.000123, 0.000000456, 0.000123456);
                    4: test_add(2.0, 3.0, 5.0);
                    5: test_add(2000.0, 300.0, 2300.0);

                    // both neg
                    6: test_add(-5.1, -3.2, -8.3);

                    // one neg
                    7: test_add(1.5, -1.25, 0.25);
                    8: test_add(1.25, -1.5, -0.25);
                    9: test_add(-1.5, 1.25, -0.25);
                    10: test_add(-1.25, 1.5, 0.25);

                    // neg to zero
                    11: test_add(1.5, -1.5, 0.0);

                    12: finish <= 1;
                endcase
            end
        end
    end
endmodule
