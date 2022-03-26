// handles adding a single bit of partial products, and returning the sum
// takes as input the carry from previous additional
// thus we do one bit per cycle, and this task represents a single cycle
//
// in this version, we try to keep the loop invariant of pos, by shifting
// operands beforehand, and padding them.
// This is mostly in preparation for trying to use full adders to maximum
// efficiency, similar to Dadda

task mul_partial_add_task(
    input [$clog2(width + 1):0] pos,
    input [width - 1:0] a,
    input [width - 1:0] b,
    input [$clog2(width):0] cin,
    output reg [bits_per_cycle - 1:0] sum,
    output reg [$clog2(width):0] cout
);
    reg rst;
    reg [width * 2 - 1:0] a_shifted;

    rst = 0;
    cout = '0;
    {cout, sum} = cin;
    `assert_known(b);
    `assert_known(pos);
    a_shifted = '0;
    a_shifted = a << (width - pos);
    for(int i = 0; i < width; i++) begin  // iterate through b
        if(b[i]) begin
            {cout, sum} = {cout, sum} + a_shifted[width - i];
        end
    end
endtask
