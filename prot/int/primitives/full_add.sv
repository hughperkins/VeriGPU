/*
synthesizes to:
- 1 x xor
- 1 x oai
- 1 x inv
- 1 x xnor
- 1 x nand

Max propagation delay: 5.2 nand units
Area:                  11.0 nand units
*/
module add(input a, input b, input cin, output sum, output cout);
    assign {cout, sum} = a + b + cin;
endmodule
