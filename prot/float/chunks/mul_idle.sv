/*
one state eval for float_mul_pipeline.sv

Propagation delay is between any pair of combinatorially connected
inputs and outputs, drawn from:
    - module inputs
    - module outputs,
    - flip-flop outputs (treated as inputs), and
    - flip-flop inputs (treated as outputs)

max propagation delay: 7.6 nand units
*/
module mul_idle(
    input req,
    input [2:0] state,
    input [float_width - 1:0] a,
    input [float_width - 1:0] b,

    output reg [2:0] n_state,
    // output reg n_a_sign,
    // output reg n_b_sign,
    output reg [float_exp_width - 1:0] n_a_exp,
    output reg [float_exp_width - 1:0] n_b_exp,
    output reg [float_mant_width:0] n_a_mant,
    output reg [float_mant_width:0] n_b_mant,
    output reg [float_mant_width * 2 + 1:0] n_new_mant,

    output reg n_new_sign,
    output reg [float_exp_width - 1:0] n_new_exp,

    output reg [float_width - 1:0] n_out,
    output reg n_ack
);
    reg [float_exp_width - 1:0] a_exp;
    reg [float_exp_width - 1:0] b_exp;
    reg [float_mant_width:0] a_mant;
    reg [float_mant_width:0] b_mant;
    reg a_sign;
    reg b_sign;

    reg rst;

    typedef enum bit[2:0] {
        IDLE,
        MUL1,
        MUL2,
        MUL3,
        S2,
        S3
    } e_state;

    always @(*) begin
        rst = 0;

        n_a_mant = '0;
        n_b_mant = '0;
        n_out = 0;
        n_ack = 0;
        n_new_mant = '0;
        n_a_exp = '0;
        n_b_exp = '0;
        // n_a_sign = 0;
        // n_b_sign = 0;
        n_state = state;

        $display("mul_idle.always");
        `assert_known(req);
        if(req) begin
            $display("mul_idle.req");
            `assert_known(a);
            `assert_known(b);

            {a_sign, n_a_exp, n_a_mant[float_mant_width - 1:0]} = a;
            {b_sign, n_b_exp, n_b_mant[float_mant_width - 1:0]} = b;

            `assert_known(n_a_exp);
            `assert_known(n_b_exp);
            if(|n_a_exp == 0 || |n_b_exp == 0) begin
                n_new_exp = '0;
                n_new_sign = 0;
                n_new_mant = '0;
                // $display("triggering out and ack for zero |a_exp=%0d b_exp=%0d", |n_a_exp, |n_b_exp);
                n_out = {n_new_sign, n_new_exp, n_new_mant[float_mant_width - 1:0]};
                n_ack = 1;
                n_state = IDLE;
            end else begin
                n_a_mant[float_mant_width] = 1;
                n_b_mant[float_mant_width] = 1;

                $display("a        %b e %0d", n_a_mant, n_a_exp);
                $display("b        %b e %0d", n_b_mant, n_a_exp);

                n_new_exp = n_a_exp + n_b_exp - 127 - float_mant_width;
                n_new_sign = a_sign ^ b_sign;

                // n_new_mant = {n_a_mant[3:0], n_a_mant, n_b_mant };
                // the multiply seems to take like 120 nand units :P
                // perhaps because it is 46-bit, and not 32-bit?
                // (when we multiply ints, we truncate out everything beyond
                // 32-bit)
                // anyway, lets split this into 2, or 3, or 23, parts
                n_new_mant = '0;
                n_state = MUL1;
            end
        end
    end
endmodule
