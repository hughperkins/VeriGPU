/*
Max propagation delay: 79.4 nand units
Area:                  353.5 nand units
*/
module sub(input [31:0] a, input [31:0] b, output [31:0] out);
    assign out = a - b;
endmodule
