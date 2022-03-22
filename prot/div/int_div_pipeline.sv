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

(we could probably do 2 bits at a time, to reduce total cycles)

required parameters:
- data_width: how many bits in the input and output ints

*/

module int_div_pipeline(
        input clk,
        input req,
        output reg ack,
        input [data_width - 1:0] a,
        input [data_width - 1:0] b,
        output reg [data_width - 1:0] quotient,
        output reg [data_width - 1:0] remainder
    );
    parameter data_width = 32;
    parameter num_regs = 32;

    parameter reg_sel_width = $clog2(num_regs);
    parameter pos_width = $clog2(data_width);

    reg [data_width - 1: 0] a_;
    reg [2 * data_width - 1: 0] shiftedb;

    reg [pos_width - 1:0] pos;
    reg run;

    reg cout;

    always @(posedge clk) begin
        if(req) begin
            {cout, pos} <= data_width - 1;
            quotient <= '0;
            a_ <= a;
            run <= 1;
            ack <= 0;
        end else if(run) begin
            if (shiftedb < {{data_width{1'b0}}, a_}) begin
                a_ <= a_ - shiftedb[data_width - 1 :0];
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

    assign shiftedb = {{data_width{1'b0}}, b} << pos;
    assign remainder = a_;
endmodule
