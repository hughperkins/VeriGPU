/*
one state eval for float_mul_pipeline.sv

from states MUL1, MUL2, etc ...

*/
module mul_idle(
    input [2:0] state,
    input [float_mant_width:0] n_a_mant,
    input [float_mant_width:0] n_b_mant,
    input [float_mant_width * 2 + 1:0] new_mant,

    output reg [2:0] n_state,
    output reg [float_mant_width * 2 + 1:0] n_new_mant
);
    parameter mul1_start = 0;
    parameter mul2_start = 8;

    typedef enum bit[2:0] {
        IDLE,
        MUL1,
        MUL2,
        MUL3,
        S2,
        S3
    } e_state;

    always @(*) begin
        n_state = state;
        n_new_mant = new_mant;

        for(int i = mul1_start; i < mul2_start; i++) begin
            partial = '0;
            partial[float_mant_width + i -: float_mant_width + 1] = n_a_mant & {float_mant_width + 1{n_b_mant[i]}};
            n_new_mant = n_new_mant + partial;
        end
        n_state = MUL2;
    end
endmodule
