module float_add(
    input [float_width - 1:0] a,
    input [float_width - 1:0] b,
    output reg [float_width -1:0] out
);
    reg rst;
    reg a_sign;
    reg [float_exp_width - 1:0] a_exp;
    reg [float_mant_width:0] a_mant;

    reg b_sign;
    reg [float_exp_width - 1:0] b_exp;
    reg [float_mant_width:0] b_mant;

    reg [float_exp_width - 1:0] new_exp;
    reg [float_mant_width + 1:0] new_mant;

    reg [float_exp_width - 1:0] exp_diff;

    // try single cycle first... see how long that takes
    always @(*) begin
        rst = 0;

        // `assert_known(a);
        // `assert_known(b);
        a_mant[float_mant_width] = 1;
        {a_sign, a_exp, a_mant[float_mant_width - 1:0]} = a;

        b_mant[float_mant_width] = 1;
        {b_sign, b_exp, b_mant[float_mant_width - 1:0]} = b;

        // $display("a sign=%0d exp=%0d mant=%0d %b", a_sign, a_exp, a_mant, a_mant);
        // $display("b sign=%0d exp=%0d mant=%0d %b", b_sign, b_exp, b_mant, b_mant);

        // `assert(a_sign == b_sign);  // for now, we don't handle different signs...

        if(a_exp > b_exp) begin
            new_exp = a_exp;
            exp_diff = a_exp - b_exp;
            b_mant = b_mant >> exp_diff;
        end else begin
            new_exp = b_exp;
            exp_diff = b_exp - a_exp;
            a_mant = a_mant >> exp_diff;
        end
        // $display("new_exp %0d a_mant %0d %b b_mant %0d %b", new_exp, a_mant, a_mant, b_mant, b_mant);
        // reg [float_exp_width - 1:0] new_exp = a_exp > b_exp ? a_exp : b_exp;
        new_mant = a_mant + b_mant;
        // $display("new_mant %0d %b", new_mant, new_mant);
        // $display("new_mant[float_mant_width + 1] %0d", new_mant[float_mant_width + 1]);
        // $display("new_mant[float_mant_width] %0d", new_mant[float_mant_width]);
        if(new_mant[float_mant_width + 1] == 1) begin
            // $display("shift mantissa right one");
            new_mant = new_mant >> 1;
            new_exp = new_exp + 1;
        end
        out = {a_sign, new_exp, new_mant[float_mant_width - 1:0]};
    end
endmodule
