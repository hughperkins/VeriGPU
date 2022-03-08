// try adding one in blocks of two bytes

module add_one_2chunks(input[31:0] pc, output [31:0] pc2);
    wire [15:0] b1, b0;
    assign {b1,b0} = pc;
    wire [15:0] b11, b01;

    wire c01; // carry from adding 1 to half n

    // results of adding cin=1 to each half
    // cn1 is then the carry out from adding 1 to half n
    // cn1 is thus the carry in for half (n + 1)
    assign {c11,b11} = b1 + 1;
    assign {c01,b01} = b0 + 1;

    wire cin1;  // carry into half n
    wire [15:0] b1out, b0out;  // actual result for each half n

    assign b0out = b01;
    // cin0 is 1
    // cin1 is result of adding 1 to half 0
    assign cin1 = c01;

    assign b1out = cin1 ? b11 : b1;

    assign pc2 = {b1out, b0out};
endmodule
