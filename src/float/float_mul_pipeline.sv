/*
multiply two floats

a, b, req, should be flip-flop inputs, change on clock tick

ack and out will be flip-flop outputs, change on clock tick
ack is set at same time as out, once result available
both will be available for a single clock tick, then we go back to idle

This version will use the generated 2bits-per-cycle int multiplier

Max propagation delay: 54.6 nand units
Area:                  3841.5 nand units

*/

`default_nettype none
module float_mul_pipeline(
    input                               clk,
    input                               rst,
    input                               req,
    output reg                          ack,
    input [float_width - 1:0]           a,
    input [float_width - 1:0]           b,
    output reg [float_width - 1:0]      out
);
    parameter bits_per_cycle = 2;

    reg                                 a_sign;
    reg [float_exp_width - 1:0]         a_exp;
    reg [float_mant_width:0]            a_mant; // [extra one][stored mantissa]

    reg                                 b_sign;
    reg [float_exp_width - 1:0]         b_exp;
    reg [float_mant_width:0]            b_mant;

    reg [float_exp_width - 1:0]         new_exp;
    reg [float_mant_width * 2 + 3:0]    new_mant;
    reg                                 new_sign;

    reg [float_exp_width - 1:0]         norm_shift;


    reg [float_width - 1:0]             n_out;
    reg                                 n_ack;

    reg [float_exp_width - 1:0]         n_a_exp;
    reg [float_mant_width:0]            n_a_mant; // [sign][overflow][extra one][stored mantissa]

    reg [float_exp_width - 1:0]         n_b_exp;
    reg [float_mant_width:0]            n_b_mant;

    reg [float_exp_width - 1:0]         n_new_exp;
    reg [float_mant_width * 2 + 3:0]    n_new_mant;

    reg                                 n_new_sign;

    reg [float_exp_width - 1:0]         n_norm_shift;

    reg [pos_width - 1:0] pos;
    reg [pos_width - 1:0] n_pos;
    reg [pos_width - 1:0] pos_plus_one;

    parameter pos_width = $clog2(float_mant_width * 2);

    typedef enum bit[1:0] {
        IDLE,
        MUL1,
        S2
    } e_state;
    reg [1:0] state;
    reg [1:0] n_state;

    reg [4:0] carry;
    reg [4:0] n_carry;

    always @(state, req, pos) begin
        // `assert_known(a);
        // `assert_known(b);

        n_state = state;

        a_sign = 0;
        b_sign = 0;

        n_a_exp = a_exp;
        n_a_mant = a_mant;
        n_b_exp = b_exp;
        n_b_mant = b_mant;
        n_new_exp = new_exp;
        n_new_mant = new_mant;
        n_new_sign = new_sign;

        n_norm_shift = norm_shift;

        n_out = '0;
        n_ack = 0;

        n_carry = carry;
        n_pos = pos;

        pos_plus_one = 0;

        `assert_known(state);
        case(state)
            IDLE: begin
                `assert_known(req);
                if(req) begin
                    // $display("floatmul.req");
                    `assert_known(a);
                    `assert_known(b);

                    n_a_mant = '0;
                    n_b_mant = '0;

                    // $display("a %b", a);
                    // $display("b %b", b);
                    {a_sign, n_a_exp, n_a_mant[float_mant_width - 1:0]} = a;
                    {b_sign, n_b_exp, n_b_mant[float_mant_width - 1:0]} = b;

                    `assert_known(n_a_exp);
                    `assert_known(n_b_exp);
                    if(|n_a_exp == 0 || |n_b_exp == 0) begin
                        n_new_exp = '0;
                        n_new_sign = 0;
                        n_new_mant = '0;
                        n_out = {n_new_sign, n_new_exp, n_new_mant[float_mant_width - 1:0]};
                        n_ack = 1;
                        n_state = IDLE;
                    end else begin
                        n_a_mant[float_mant_width] = 1;
                        n_b_mant[float_mant_width] = 1;

                        // $display("a        %b e %0d", n_a_mant, n_a_exp);
                        // $display("b        %b e %0d", n_b_mant, n_a_exp);

                        n_new_exp = n_a_exp + n_b_exp - 127 - float_mant_width;
                        n_new_sign = a_sign ^ b_sign;

                        n_new_mant = '0;
                        n_pos = 0;
                        n_carry = 0;
                        n_state = MUL1;
                        n_norm_shift = 0;
                    end
                end
            end
            MUL1: begin
                mul_pipeline_cycle_24bit_2bpc(
                    pos,
                    n_a_mant,
                    n_b_mant,
                    carry,
                    n_new_mant[pos + bits_per_cycle - 1 -: bits_per_cycle],
                    n_carry
                );
                // $display("pos=%0d n_new_mant=%b carry=%b n_carry=%b norm_shift %0d", n_pos, n_new_mant, carry, n_carry, norm_shift);
                `assert_known(n_pos);
                if(pos >= float_mant_width * 2 + 1) begin
                    n_state = S2;
                end
                n_pos = pos + 2;
                pos_plus_one = pos + 1;
                if(n_new_mant[pos_plus_one]) begin
                    // n_norm_shift = pos +  { {pos_width - 1{1'b0}}, 1'b1 };
                    n_norm_shift = '0;
                    n_norm_shift[pos_width - 1:0] = pos_plus_one;
                end else if(n_new_mant[pos]) begin
                    n_norm_shift = '0;
                    n_norm_shift[pos_width - 1:0] = pos;
                end
                // $display("MUL1 pos_plus_one=%0d n_new_mant[pos_plus_one] %b, n_norm_shift %0d", pos_plus_one, n_new_mant[pos_plus_one], n_norm_shift);
            end
            S2: begin
                // $display("floatmul.S3");
                // $display("S3 n_norm_shift %0d", n_norm_shift);
                n_norm_shift = n_norm_shift - float_mant_width;
                n_new_mant = n_new_mant >> n_norm_shift;
                n_new_exp = n_new_exp + n_norm_shift;

                // $display("floatmul.S2b n_new_mant=%b n_new_exp=%0d", n_new_mant, n_new_exp);

                n_out = {n_new_sign, n_new_exp, n_new_mant[float_mant_width - 1:0]};
                n_ack = 1;
                n_state = IDLE;
            end
            default: begin
            end
        endcase
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            state <= IDLE;

            out <= 0;
            ack <= 0;

            a_exp <= '0;
            a_mant <= '0;

            b_exp <= '0;
            b_mant <= '0;

            new_exp <= '0;
            new_mant <= '0;
            new_sign <= '0;

            norm_shift <= '0;

            carry <= '0;
            pos <= '0;
        end else begin
            out <= n_out;
            ack <= n_ack;

            state <= n_state;

            a_exp <= n_a_exp;
            a_mant <= n_a_mant;

            b_exp <= n_b_exp;
            b_mant <= n_b_mant;

            new_exp <= n_new_exp;
            new_mant <= n_new_mant;
            new_sign <= n_new_sign;

            norm_shift <= n_norm_shift;
            // $display("mul posedge norm_shift=%0d", norm_shift);
            // $display("float_mul_pipeline.ff state=%0d pos=%0d n_new_mant=%b out=%b ack=%b", state, pos, n_new_mant, out, ack);

            carry <= n_carry;
            pos <= n_pos;
        end
    end
endmodule
