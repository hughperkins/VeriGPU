/*
Add two floats

Currently a bit long... :

Propagation delay is between any pair of combinatorially connected
inputs and outputs, drawn from:
    - module inputs
    - module outputs,
    - flip-flop outputs (treated as inputs), and
    - flip-flop inputs (treated as outputs)

max propagation delay: 149.6 nand units
*/

module float_add(
    input [float_width - 1:0] a,
    input [float_width - 1:0] b,
    output reg [float_width -1:0] out
);
    reg rst;
    reg a_sign;
    reg [float_exp_width - 1:0] a_exp;
    reg [float_mant_width + 2:0] a_mant; // [sign][overflow][extra one][stored mantissa]

    reg b_sign;
    reg [float_exp_width - 1:0] b_exp;
    reg [float_mant_width + 2:0] b_mant;

    reg [float_exp_width - 1:0] new_exp;
    reg [float_mant_width + 2:0] new_mant;

    reg [float_exp_width - 1:0] exp_diff;

    reg [float_exp_width - 1:0] exp_norm;

    reg [float_mant_width + 1:0] new_mant_lookup[float_mant_width];
    reg [float_exp_width - 1:0] norm_shift;

    reg new_sign;

    // try single cycle first... see how long that takes
    always @(*) begin
        rst = 0;

        // `assert_known(a);
        // `assert_known(b);
        a_mant[float_mant_width + 2:float_mant_width] = 3'b001;
        {a_sign, a_exp, a_mant[float_mant_width - 1:0]} = a;

        b_mant[float_mant_width + 2:float_mant_width] = 3'b001;
        {b_sign, b_exp, b_mant[float_mant_width - 1:0]} = b;

        if(a_sign != b_sign) begin
            if(a_sign == 1) begin
                $display("inverting a_mant before=%b", a_mant);
                a_mant = ~a_mant;
                $display("inverting a_mant after= %b", a_mant);
                a_mant = a_mant + 1;
                $display("inverting a_mant after= %b", a_mant);
                new_sign = 0;
            end else begin
                $display("inverting b_mant before=%b", b_mant);
                b_mant = ~b_mant;
                $display("inverting b_mant after= %b", b_mant);
                b_mant = b_mant + 1;
                $display("inverting b_mant after= %b", b_mant);
                new_sign = 0;
            end
        end else begin
            new_sign = a_sign;
        end

        if(a_exp > b_exp) begin
            new_exp = a_exp;
            exp_diff = a_exp - b_exp;
            b_mant = b_mant >> exp_diff;
        end else begin
            new_exp = b_exp;
            exp_diff = b_exp - a_exp;
            a_mant = a_mant >> exp_diff;
        end
        $display("new_exp %0d a_mant %0d %b b_mant %0d %b", new_exp, a_mant, a_mant, b_mant, b_mant);
        new_mant = a_mant + b_mant;
        $display("a        %b", a_mant);
        $display("b        %b", b_mant);
        $display("new_mant %b", new_mant);
        // flip sign if necessary
        if(new_mant[float_mant_width + 2]) begin
            // its negative
            new_sign = ~new_sign;
            new_mant = ~new_mant + 1;
            $display("new mant neg, flip sign");
            $display("new_mant %b", new_mant);
        end
        if(new_mant[float_mant_width + 1] == 1) begin
            new_mant[float_mant_width + 1:0] = new_mant[float_mant_width + 1:0] >> 1;
            new_exp = new_exp + 1;
        end
        $display("new_mant %b", new_mant);
        norm_shift = 0;
        if(|new_mant == 0) begin
            // if eveyrthing is zero ,then just return zero
            new_exp = '0;
            a_sign = 0;
        end else begin
            for(int shift = float_mant_width - 1; shift >= 0; shift--) begin
                $display("shift %0d new_mant[shift]=%0d", shift, new_mant[float_mant_width - shift]);
                if(new_mant[float_mant_width - shift] == 1) begin
                    norm_shift = shift;
                    new_mant_lookup[shift] = new_mant << shift;
                    $display("    new_mant_lookup[shift]=%b norm_shift=%0d", new_mant_lookup[shift], norm_shift);
                end else begin
                    new_mant_lookup[shift] = 0;
                end
            end
            new_mant = new_mant_lookup[norm_shift];
            new_exp = new_exp - norm_shift;
        end
        $display("new_mant %b new_exp %0d", new_mant, new_exp);
        out = {new_sign, new_exp, new_mant[float_mant_width - 1:0]};
    end
endmodule
