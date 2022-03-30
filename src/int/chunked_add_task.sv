/*
try adding one in blocks of two bytes

Max propagation delay: 43.8 nand units
Area:                  357.5 nand units
*/
task chunked_add_task(
    input [adder_width - 1:0] a,
    input [adder_width - 1:0] b,
    output reg [adder_width - 1:0] out
);
    parameter adder_width = 32;
    parameter half_width = adder_width / 2;

    reg [half_width - 1:0] a1, a0;
    reg [half_width - 1:0] b1, b0;

    reg [half_width - 1:0] out0;
    reg [half_width - 1:0] out10, out11;
    reg [half_width - 1:0] out1;

    reg c0;

    reg [half_width - 1:0] b1out, b0out;  // actual result for each half n

    {a1, a0} = a;
    {b1, b0} = b;

    {c0, out0} = a0 + b0;

    out10 = a1 + b1;
    out11 = a1 + b1 + 1;

    out1 = c0 ? out11 : out10;

    out = {out1, out0};
endtask
