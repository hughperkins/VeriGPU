/*
multiply two floats

a, b, req, should be flip-flop inputs, change on clock tick

ack and out will be flip-flop outputs, change on clock tick
ack is set at same time as out, once result available
both will be available for a single clock tick, then we go back to idle

Max propagation delay: 78.4 nand units
Area:                  9392.0 nand units
*/
module float_mul_pipeline(
    input                               clk,
    input                               rst,
    input                               req,
    output reg                          ack,
    input [float_width - 1:0]           a,
    input [float_width - 1:0]           b,
    output reg [float_width - 1:0]      out
);
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
    reg [float_mant_width * 2 + 1:0]    n_new_mant;

    reg                                 n_new_sign;

    reg [float_exp_width - 1:0]         n_norm_shift;

    reg [float_mant_width * 2 + 1:0]    partial;

    typedef enum bit[2:0] {
        IDLE,
        MUL1,
        MUL2,
        MUL3,
        S2,
        S3
    } e_state;
    reg [2:0] state;
    reg [2:0] n_state;

    // bit positions for each chunk of multiplication
    // there are float_mant_width + 1 bits altogether (24 for single)
    // (remember there is an extra 1 that isnt stored inside the float
    // mantissa representation)
    // we want these evenly split
    // the names correspond to the state names above, but MUL1
    // state is implicitly IDLE, when we receive a req
    // each chunk finishes where the next one begins
    parameter mul1_start = 0;
    parameter mul2_start = 8;
    parameter mul3_start = 16;
    parameter mul4_start = 24;

    reg [5:0] carry;
    reg [5:0] n_carry;

    always @(state, req) begin
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

        $display("%0d float_mul.always state=%0d", $time, state);
        `assert_known(state);
        case(state)
            IDLE: begin
                `assert_known(req);
                if(req) begin
                    $display("floatmul.req");
                    `assert_known(a);
                    `assert_known(b);

                    n_a_mant = '0;
                    n_b_mant = '0;

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
                        n_carry = '0;
                        for(int i = 0; i < float_mant_width + 1; i++) begin
                            partial = '0;
                            partial[float_mant_width + i -: float_mant_width + 1] = n_a_mant & {float_mant_width + 1{n_b_mant[i]}};
                            {n_carry, n_new_mant[19:0]} = {n_carry, n_new_mant[19:0]} + partial[19:0];
                        end
                        $display("n_new_mant=%b partial=%b carry=%b", n_new_mant, partial, n_carry);
                        n_state = MUL2;
                    end
                end
            end
            MUL2: begin
                
                n_new_mant = n_new_mant + {n_carry, {20{1'b0}} };
                n_carry = '0;
                for(int i = 0; i < float_mant_width + 1; i++) begin
                    partial = '0;
                    partial[float_mant_width + i -: float_mant_width + 1] = n_a_mant & {float_mant_width + 1{n_b_mant[i]}};
                    {n_carry, n_new_mant[27:20]} = {n_carry,n_new_mant[27:20]} + partial[27:20];
                    // $display("i=%2d n_new_mant=%b partial=%b", i, n_new_mant, partial);
                end
                $display("n_new_mant=%b partial=%b carry=%b", n_new_mant, partial, n_carry);
                
                n_state = MUL3;
            end
            MUL3: begin
                
                n_new_mant = n_new_mant + {n_carry, {28{1'b0}} };
                n_carry = '0;
                for(int i = 0; i < float_mant_width + 1; i++) begin
                    partial = '0;
                    partial[float_mant_width + i -: float_mant_width + 1] = n_a_mant & {float_mant_width + 1{n_b_mant[i]}};
                    n_new_mant[47:28] = n_new_mant[47:28] + partial[47:28];
                    // $display("i=%2d n_new_mant=%b partial=%b", i, n_new_mant, partial);
                end
                $display("n_new_mant=%b partial=%b carry=%b", n_new_mant, partial, n_carry);
                
                n_state = S2;
            end
            S2: begin
                $display("floatmul.S2");
                for(int shift = float_mant_width + 1; shift <= 2 * float_mant_width + 1; shift++) begin
                    if(n_new_mant[shift] == 1) begin
                        n_norm_shift = shift;
                    end
                end
                $display("n_norm_shift %0d", n_norm_shift);
                n_state = S3;
            end
            S3: begin
                $display("floatmul.S3");
                $display("n_norm_shift %0d", n_norm_shift);
                // $display("norm_shift=%0d", norm_shift);
                n_norm_shift = n_norm_shift - float_mant_width;
                n_new_mant = n_new_mant >> n_norm_shift;
                n_new_exp = n_new_exp + n_norm_shift;

                // $display("new_mant %b new_exp %0d", new_mant, new_exp);
                // $display("triggering out and ack");
                n_out = {n_new_sign, n_new_exp, n_new_mant[float_mant_width - 1:0]};
                n_ack = 1;
                n_state = IDLE;
            end
            default: begin
                // `assert(0);  // should never get here
            end
        endcase
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            state <= IDLE;

            out <= 0;
            ack <= 0;

            a_sign <= 0;
            a_exp <= '0;
            a_mant <= '0;

            b_sign <= 0;
            b_exp <= '0;
            b_mant <= '0;

            new_exp <= '0;
            new_mant <= '0;
            new_sign <= '0;

            norm_shift <= '0;

            carry <= '0;
        end else begin
            // $display("float_add_pipeline not rst state=%0d", n_state);
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

            carry <= n_carry;
        end
    end
endmodule
