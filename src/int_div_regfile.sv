/*
use pipeline for int_div, but test writin back to a registry file

required parameters:
- data_width: how many bits in the input and output ints
- num_regs: how many registers there are in the registry file


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
        output reg rf_wr_req
);
    parameter data_width = 32;
    parameter num_regs = 32;

    parameter reg_sel_width = $clog2(num_regs);
    parameter pos_width = $clog2(data_width);
    parameter data_width_minus_1 = data_width - 1;

    reg [data_width - 1:0] quotient;
    reg [data_width - 1:0] remainder;
    reg [data_width - 1: 0] a_;
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

    always @(posedge clk, posedge rst) begin
        // $strobe("always %0d req=%0b rst=%0b", state, req, rst);
        if (rst) begin
            busy <= 0;
            rf_wr_req <= 0;
            state <= IDLE;
        end else begin
            rf_wr_req <= 0;
            // $strobe("state %0d req=%0b", state, req);
            case(state)
                IDLE: begin
                    if(req) begin
                        pos <= data_width_minus_1;
                        quotient <= '0;
                        a_ <= a;
                        busy <= 1;
                        state <= CALC;
                        wrote_quotient <= 0;
                        wrote_modulus <= 0;
                    end
                end
                CALC: begin
                    if (shiftedb < {{data_width{1'b0}}, a_}) begin
                        a_ <= a_ - shiftedb[data_width - 1 :0];
                        quotient[pos] <= 1;
                    end
                    if (pos == 0) begin
                        state <= WRITE;
                    end else begin
                        pos <= pos - 1;
                    end
                end
                WRITE: begin
                    if(r_quot_sel != 0 && ~wrote_quotient) begin
                        rf_wr_req <= 1;
                        rf_wr_sel <= r_quot_sel;
                        rf_wr_data <= quotient;
                        wrote_quotient <= 1;
                    end else if(r_mod_sel != 0 && ~wrote_modulus) begin
                        rf_wr_req <= 1;
                        rf_wr_sel <= r_mod_sel;
                        rf_wr_data <= remainder;
                        wrote_modulus <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end else begin
                        busy <= 0;
                        state <= IDLE;
                        rf_wr_req <= 0;
                    end
                end
            endcase
        end
    end

    assign shiftedb = {{data_width{1'b0}}, b} << pos;
    assign remainder = a_;
endmodule
