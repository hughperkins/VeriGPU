/*
use pipeline for int_div, but test writin back to a registry file

required parameters:
- data_width: how many bits in the input and output ints
- num_regs: how many registers there are in the registry file

protocol:
- client wants to divide a by b; and put quotient into register r_quot_sel,
  and modulus into r_mod_sel
- client sets these lines to desired values, and sets req to 1, waits for clock tick
- after clock tick, client sets req to 0
- module takes over
- each clock tick works way through the division
- once done, writes the results to the registers, i.e. quotient then modulus
     - if a register selector is 0, it is ignored,
     - otherwise, for each register, one at a time (different clock ticks):
         - puts selector and data onto rf_wr_sel and rf_wr_data, and sets rf_wr_req; then waits for rf_wr_ack to turn high
         - once rf_wr_ack is high, sets rf_wr_req back to 0
         - if required, continues with modulus, or moves back to IDLE state

$ python toy_proc/timing.py --in-verilog src/const.sv src/int_div_regfile.sv

Propagation delay is between any pair of combinatorially connected
inputs and outputs, drawn from:
    - module inputs
    - module outputs,
    - flip-flop outputs (treated as inputs), and
    - flip-flop inputs (treated as outputs)

max propagation delay: 67.0 nand units
*/

module int_div_regfile(
        input clk,
        input rst,

        input req,
        output reg busy,

        input [reg_sel_width - 1: 0] r_quot_sel,  // 0 means, dont write (i.e. x0)
        input [reg_sel_width - 1: 0] r_mod_sel,   // 0 means, dont write  (i.e. x0)
        input [data_width - 1:0] a,
        input [data_width - 1:0] b,

        output reg [reg_sel_width - 1:0] rf_wr_sel,
        output reg [data_width - 1:0] rf_wr_data,
        output reg rf_wr_req,
        input rf_wr_ack
);
    parameter data_width = 32;
    parameter num_regs = 32;

    parameter reg_sel_width = $clog2(num_regs);
    parameter pos_width = $clog2(data_width);
    parameter data_width_minus_1 = data_width - 1;

    reg [data_width - 1:0] quotient;
    reg [data_width - 1:0] remainder;
    reg [data_width - 1: 0] a_remaining;
    reg [2 * data_width - 1: 0] shiftedb;

    reg [pos_width - 1:0] pos;

    reg wrote_quotient;
    reg wrote_modulus;

    typedef enum {
        IDLE,
        CALC,
        WRITE
    } e_state;

    reg [1:0] state;

    reg [1:0] next_state;
    reg [pos_width -1:0] next_pos;
    reg next_wrote_quotient;
    reg next_wrote_modulus;
    reg [data_width -1:0] next_quotient;
    reg [data_width -1:0] next_a_remaining;

    always @(*) begin
        rf_wr_req = 0;
        next_pos = pos;
        busy = 0;
        next_quotient = quotient;
        next_wrote_modulus = wrote_modulus;
        next_wrote_quotient = wrote_quotient;
        next_a_remaining = a_remaining;

        shiftedb = {{data_width{1'b0}}, b} << pos;

        // $strobe("state %0d req=%0b", state, req);
        case(state)
            IDLE: begin
                if(req) begin
                    next_pos = data_width_minus_1;
                    next_quotient = '0;
                    next_a_remaining = a;
                    busy = 1;
                    next_state = CALC;
                    next_wrote_quotient = 0;
                    next_wrote_modulus = 0;
                end
            end
            CALC: begin
                busy = 1;
                if (shiftedb < {{data_width{1'b0}}, next_a_remaining}) begin
                    next_a_remaining = next_a_remaining - shiftedb[data_width - 1 :0];
                    next_quotient[pos] = 1;
                end
                if (pos == 0) begin
                    next_state = WRITE;
                end else begin
                    next_pos = pos - 1;
                end
            end
            WRITE: begin
                busy = 1;
                remainder = next_a_remaining;
                if(r_quot_sel != 0 && ~wrote_quotient) begin
                    rf_wr_req = 1;
                    rf_wr_sel = r_quot_sel;
                    rf_wr_data = quotient;
                    next_wrote_quotient = 1;
                end else if(r_mod_sel != 0 && ~wrote_modulus) begin
                    rf_wr_req = 1;
                    rf_wr_sel = r_mod_sel;
                    rf_wr_data = remainder;
                    next_wrote_modulus = 1;
                    busy = 0;
                    next_state = IDLE;
                end else begin
                    busy = 0;
                    next_state = IDLE;
                    rf_wr_req = 0;
                end
            end
        endcase
    end

    always @(posedge clk, posedge rst) begin
        // $strobe("always %0d req=%0b rst=%0b", state, req, rst);
        if (rst) begin
            state <= IDLE;
            pos <= 0;
        end else begin
            state <= next_state;
            pos <= next_pos;
            quotient <= next_quotient;
            a_remaining <= next_a_remaining;
            wrote_modulus <= next_wrote_modulus;
            wrote_quotient <= next_wrote_quotient;
        end
    end
endmodule
