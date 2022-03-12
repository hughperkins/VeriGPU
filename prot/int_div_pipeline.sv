/*
attempt to use pipeline for int_div

timing, for bitwidth = 32, poswidth = 5:

$ python toy_proc/timing.py --in-verilog prot/int_div_pipeline.sv 

Propagation delay is between any pair of combinatorially connected
inputs and outputs, drawn from:
    - module inputs
    - module outputs,
    - flip-flop outputs (treated as inputs), and
    - flip-flop inputs (treated as outputs)

max propagation delay: 38.6 nand units

required parameters:
- bitwidth: how many bits in the input and output ints

*/

module int_div_pipeline(input clk, input req, output reg ack, input [bitwidth - 1:0] a, input [bitwidth - 1:0] b, output reg [bitwidth - 1:0] quotient, output reg [bitwidth - 1:0] remainder);
    parameter bitwidth = 32;

    parameter poswidth = $clog2(bitwidth);

    reg [bitwidth - 1:0] result1[bitwidth];
    reg [bitwidth - 1:0] result2[bitwidth];

    reg [bitwidth - 1: 0] a_;
    reg [2 * bitwidth - 1: 0] shiftedb;

    reg [poswidth - 1:0] pos;
    reg run;

    reg cout;

    always @(posedge clk) begin
        if(req) begin
            {cout, pos} <= bitwidth - 1;
            quotient <= '0;
            a_ <= a;
            run <= 1;
            ack <= 0;
        end else if(run) begin
            if (shiftedb < {{bitwidth{1'b0}}, a_}) begin
                a_ <= a_ - shiftedb[bitwidth - 1 :0];
                quotient[pos] <= 1;
            end
            if (pos == 0) begin
                ack <= 1;
                run <= 0;
            end else begin
                pos <= pos - 1;
            end
        end else begin
            ack <= 0;
        end
    end

    assign shiftedb = {{bitwidth{1'b0}}, b} << pos;
    assign remainder = a_;
endmodule
