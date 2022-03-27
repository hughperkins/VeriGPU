/*
5 x BUF
1 x NOR
1 x AOI21X1
1 x AOI22X1
1 x NAND3X1
1 x INVX1
1 x NOR2X1

5 x buf
1 x nor
1 x aoi2
1 x nand3
1 x inv
1 x nor
1 x aoi

Max propagation delay: 6.6 nand units
Area:                  9.0 nand units
*/
task full_adder(
    input a,
    input b,
    input cin,
    output reg sum,
    output reg cout
);
    cout = a ^ b ^ cin;
    sum = (a & b) | (cin & (a | b));
endtask
