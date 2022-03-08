// try adding one in blocks of one byte

/*
function add_one(input [31:0] pc, output [31:0] pc2);
    // pc2 = pc + 1;
    wire [7:0] b1, b2, b3, b4;
    wire {b3,b2,b1,b0} = pc;
    // wire [7:0] b10, b20, b30, b40;
    wire [7:0] b31, b21, b11, b01;

    wire c21, c11, c01; // carry from adding 1 to byte n

    // results of adding cin=1 to each byte
    // cn1 is then the carry out from adding 1 to byte n
    // cn1 is thus the carry in for byte (n + 1)
    assign b31 = b3 + 1;
    assign {c21,b21} = b2 + 1;
    assign {c11,b11} = b1 + 1;
    assign {c01,b01} = b0 + 1;

    wire cin3, cin2, cin1;  // carry into byte n
    wire b3out, b2out, b1out, b0out;  // actual result for each byte n

    assign b0out = b01;
    // cin0 is 1
    // cin1 is result of adding 1 to byte 0
    assign cin1 = c01;
    // cin2 is either 0, or result of adding 1 to byte 1
    // we are only going to add 1 to byte 1, if cin1 is 1
    assign cin2 = cin1 ? c11 : 0;
    assign cin3 = cin2 ? c21 : 0;
    assign cin4 = cin3 ? c31 : 0;

    assign b1out = cin1 ? b11 : b1;
    assign b2out = cin2 ? b21 : b2;
    assign b3out = cin3 ? b31 : 3;

    assign pc2 = {b3out,b2out,b1out,b0out};
endfunction
*/

module add_one_chunked(input[31:0] pc, output [31:0] pc2);
    // pc2 = pc + 1;
    wire [7:0] b3, b2, b1, b0;
    assign {b3,b2,b1,b0} = pc;
    // wire [7:0] b10, b20, b30, b40;
    wire [7:0] b31, b21, b11, b01;

    wire c21, c11, c01; // carry from adding 1 to byte n

    // results of adding cin=1 to each byte
    // cn1 is then the carry out from adding 1 to byte n
    // cn1 is thus the carry in for byte (n + 1)
    assign b31 = b3 + 1;
    assign {c21,b21} = b2 + 1;
    assign {c11,b11} = b1 + 1;
    assign {c01,b01} = b0 + 1;

    wire cin3, cin2, cin1;  // carry into byte n
    wire [7:0] b3out, b2out, b1out, b0out;  // actual result for each byte n

    assign b0out = b01;
    // cin0 is 1
    // cin1 is result of adding 1 to byte 0
    assign cin1 = c01;
    // cin2 is either 0, or result of adding 1 to byte 1
    // we are only going to add 1 to byte 1, if cin1 is 1
    assign cin2 = cin1 ? c11 : 0;
    assign cin3 = cin2 ? c21 : 0;
    // assign cin4 = cin3 ? c31 : 0;

    assign b1out = cin1 ? b11 : b1;
    assign b2out = cin2 ? b21 : b2;
    assign b3out = cin3 ? b31 : b3;

    // assign pc2 = {b3out,b2out,b1out,b0out};
    // assign pc2 = {b3,b2,b1,b0};
    // assign pc2 = {{24{1'b0}},b0out};
    // assign pc2 = {{16{1'b0}},b1out, b0out};
    // assign pc2 = {{8{1'b0}},b2out, b1out, b0out};
    assign pc2 = {b3out,b2out, b1out, b0out};
endmodule
