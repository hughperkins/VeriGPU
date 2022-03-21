/*
Add two floats

Propagation delay is between any pair of combinatorially connected
inputs and outputs, drawn from:
    - module inputs
    - module outputs,
    - flip-flop outputs (treated as inputs), and
    - flip-flop inputs (treated as outputs)

max propagation delay: 132.8 nand units
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

        // this if...else block add 40 nand unit delay
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

        // this if-else block adds 56 nand units
        if(a_sign != b_sign) begin
            if(a_mant > b_mant) begin
                // a bigger, sign comes from a,
                // subract b from a
                new_mant = a_mant - b_mant;
                new_sign = a_sign;
            end else begin
                // b bigger
                // take sign from a, subtract a from b
                new_mant = b_mant - a_mant;
                new_sign = b_sign;
            end
        end else begin
            new_sign = a_sign;
            new_mant = a_mant + b_mant;
        end

        $display("a        %b", a_mant);
        $display("b        %b", b_mant);
        $display("new_mant %b", new_mant);
        // this if adds 8 nand units delay
        if(new_mant[float_mant_width + 1] == 1) begin
            new_mant[float_mant_width + 1:0] = new_mant[float_mant_width + 1:0] >> 1;
            new_exp = new_exp + 1;
        end
        $display("new_mant %b", new_mant);
        norm_shift = 0;
        // this if-else adds 35 nand units delay
        if(|new_mant == 0) begin
            // if eveyrthing is zero ,then just return zero
            $display("all zero");
            new_exp = '0;
            new_sign = 0;
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
