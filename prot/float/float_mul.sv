module float_mul(
    input [float_width - 1:0]       a,
    input [float_width - 1:0]       b,
    output reg [float_width - 1:0]  out
);
    reg rst;

    reg a_sign;
    reg [float_exp_width - 1:0] a_exp;
    reg [float_mant_width:0] a_mant; // [sign][overflow][extra one][stored mantissa]

    reg b_sign;
    reg [float_exp_width - 1:0] b_exp;
    reg [float_mant_width:0] b_mant;

    reg [float_exp_width - 1:0] new_exp;
    reg [float_mant_width * 2 + 3:0] new_mant;
    reg new_sign;

    reg [float_mant_width - 1:0] new_mant_lookup[float_mant_width + 2];
    reg [float_exp_width - 1:0] norm_shift;

    always @(a, b) begin
        rst = 0;

        // `assert_known(a);
        // `assert_known(b);

        a_mant = '0;
        b_mant = '0;

        {a_sign, a_exp, a_mant[float_mant_width - 1:0]} = a;
        {b_sign, b_exp, b_mant[float_mant_width - 1:0]} = b;

        if(|a_exp == 0 || b_exp == 0) begin
            new_exp = '0;
            new_sign = 0;
            new_mant = '0;
        end else begin
            a_mant[float_mant_width] = 1;
            b_mant[float_mant_width] = 1;

            $display("a        %b e %0d", a_mant, a_exp);
            $display("b        %b e %0d", b_mant, a_exp);            // $display("new_mant %b", new_mant);

            new_mant = a_mant * b_mant;
            new_exp = a_exp + b_exp - 127 - float_mant_width;

            $display("new_mant %b new_exp %0d", new_mant, new_exp);

            for(int shift = 1; shift <= float_mant_width + 1; shift++) begin
                $display("shift %0d new_mant[shift]=%0d", shift, new_mant[float_mant_width - shift]);
                if(new_mant[float_mant_width + shift] == 1) begin
                    norm_shift = shift;
                    $display("new_mant >> shift %b", new_mant >> shift);
                    new_mant_lookup[shift] = (new_mant >> shift);
                    $display("    new_mant_lookup[shift]=%b norm_shift=%0d", new_mant_lookup[shift], norm_shift);
                end else begin
                    new_mant_lookup[shift] = 0;
                end
            end
            $display("norm_shift=%0d", norm_shift);
            new_mant = new_mant_lookup[norm_shift];
            new_exp = new_exp + norm_shift;

            new_sign = a_sign ^ b_sign;
        end
        $display("new_mant %b new_exp %0d", new_mant, new_exp);
        out = {new_sign, new_exp, new_mant[float_mant_width - 1:0]};
    end
endmodule
