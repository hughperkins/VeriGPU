/*
Max propagation delay: 13.6 nand units
Area:                  373.0 nand units
*/
module shiftleft(
    input [31:0] a,
    input [5:0] b,
    output [31:0] out
);
    assign out = a << b;
endmodule
