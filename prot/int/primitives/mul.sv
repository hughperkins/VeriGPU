/*
multiplication module, to measure nand unit propagation delay

Max propagation delay: 82.8 nand units
Area:                  5369.5 nand units
*/
module mul(input [31:0] a, input [31:0] b, output [31:0] out);
    assign out = a * b;
endmodule
