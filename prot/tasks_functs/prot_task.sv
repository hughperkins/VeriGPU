// see if yosys can handle this
task prot_task(
    input [1:0] a,
    input [1:0] b,
    output reg [1:0] out
);
    out = a + b;
endtask
