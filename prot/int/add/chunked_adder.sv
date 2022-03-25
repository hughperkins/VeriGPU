// try adding one in blocks of two bytes

module chunked_adder(input[adder_width - 1:0] a, [adder_width - 1:0] b, output [adder_width - 1:0] y);
    parameter adder_width = 32;
    parameter half_width = adder_width / 2;

    wire [half_width - 1:0] a1, a0;
    assign {a1,a0} = a;

    wire [half_width - 1:0] b1, b0;
    assign {b1,b0} = b;

    // wire [15:0] a11, a01;
    wire [half_width - 1:0] y00;
    wire [half_width - 1:0] y11, y01;

    wire c01; // carry from adding 1 to half n

    // results of adding cin=1 to each half
    // cn1 is then the carry out from adding 1 to half n
    // cn1 is thus the carry in for half (n + 1)
    assign {c11,y11} = a1 + b1 + 1;
    assign {c10,y10} = a1 + b1;
    assign {c00,y00} = a0 + b0;

    wire cin1;  // carry into half n
    wire [half_width - 1:0] b1out, b0out;  // actual result for each half n

    assign b0out = b01;
    // cin0 is 1
    // cin1 is result of adding 1 to half 0
    assign cin1 = c01;

    assign b1out = cin1 ? b11 : b1;

    assign pc2 = {b1out, b0out};
endmodule
