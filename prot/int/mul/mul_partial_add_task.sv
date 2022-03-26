// handles adding a single bit of partial products, and returning the sum
// takes as input the carry from previous additional
// thus we do one bit per cycle, and this task represents a single cycle
// parameter width = 24;
// parameter bits_per_cycle = 1;

task mul_partial_add_task(
    input [$clog2(width + 1):0] pos,
    input [width - 1:0] a,
    input [width - 1:0] b,
    input [$clog2(width):0] cin,
    output reg [bits_per_cycle - 1:0] sum,
    output reg [$clog2(width):0] cout
);
    reg rst;
    rst = 0;
    /*
    a3 a2 a1 a0
    b3 b2 b1 b0

                          pos=2
                     a3b0 a2b0 a1b0 a0b0
                a3b1 a2b1 a1b1 a0b1
           a3b2 a2b2 a1b2 a0b2
      a3b3 a2b3 a1b3 a0b3
    */
    cout = '0;
    {cout, sum} = cin;
    // {cout, sum} = '0;
    `assert_known(b);
    `assert_known(pos);
    // $display("pos=%0d", pos);
    for(int i = 0; i < width; i++) begin
        if(b[i]) begin
            if(pos >= i && pos - i < width) begin
                {cout, sum} = {cout, sum} + a[pos - i];
            end
        end
    end
endtask
