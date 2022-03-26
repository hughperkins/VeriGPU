/*
division module, to measure nand unit propagation delay

Max propagation delay: 1215.8 nand units
Area:                  7594.0 nand units
*/
module div(input [31:0] a, input [31:0] b, output [31:0] out);
    assign out = a / b;
endmodule
