/*
Max propagation delay: 2.6 nand units
Area:                  4.5 nand units
*/
task half_adder(
    input a,
    input b,
    output reg sum,
    output reg cout
);
    cout = a ^ b;
    sum = (a & b);
endtask
