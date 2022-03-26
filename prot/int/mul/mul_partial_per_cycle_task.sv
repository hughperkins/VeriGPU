/*
Max propagation delay: 97.8 nand units
Area:                  2103.0 nand units
*/
task mul_partial_per_cycle_task(
    input [$clog2(width * 2) - 1:0] pos,
    input [width - 1:0] a,
    input [width - 1:0] b,
    input [width * 2 - 1:0] prev_out,
    output reg [width * 2 - 1:0] out
);
    out = prev_out;
    out[width + pos - 1 -: width] = out[width + pos - 1 -: width] + (a & {width{b[pos]}});
endtask
