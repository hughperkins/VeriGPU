/*
For 32-bit data width:
Max propagation delay: 82.8 nand units
Area:                  5371.5 nand units

(interestingly, this propagation delay is identical to yosys implementation of 32-bit `*`)
*/
module mul(
    input [data_width - 1:0] a,
    input [data_width - 1:0] b,
    output reg [data_width * 2 - 1:0] out
);
    reg [data_width * 2 - 1:0] out_wide;
    reg [data_width * 2 - 1:0] partial;
    always @(a, b) begin
        out_wide = '0;
        // i here represents which bit from the right of b, we 
        // are forming a partial product with a with
        for(int i = 0; i < data_width; i++) begin
            partial = '0;
            // $display("a & b[i]", a & b[i]);
            partial[data_width + i - 1 -: data_width ] = a & {data_width{b[i]}};
            // $display("out_wide %b partial %b", out_wide, partial);
            out_wide = out_wide + partial;
        end
        out = out_wide[data_width - 1:0];
    end
endmodule
