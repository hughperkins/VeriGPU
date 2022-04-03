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

$ python verigpu/timing.py --in-verilog src/int_div_regfile.sv 

Propagation delay is between any pair of combinatorially connected
inputs and outputs, drawn from:
    - module inputs
    - module outputs,
    - flip-flop outputs (treated as inputs), and
    - flip-flop inputs (treated as outputs)

max propagation delay: 44.2 nand units
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
    parameter data_width_minus_1 = (data_width - 1);

    parameter state_width = 3;
    typedef enum bit[state_width - 1:0] {
        IDLE,
        CALC,
        WRITE,
        WRITING_QUOTIENT,
        WRITING_MODULUS
    } e_state;

    reg [state_width - 1:0] state;

    reg [reg_sel_width - 1:0] internal_r_quot_sel;
    reg [reg_sel_width - 1:0] internal_r_mod_sel;
    // reg [data_width -1:0] internal_b;
    reg [data_width * 2 - 1:0] b_shifted;

    reg [pos_width - 1:0] pos;
    reg [data_width - 1:0] quotient;
    reg [data_width - 1:0] remainder;
    reg [data_width - 1: 0] a_remaining;

    reg [reg_sel_width - 1:0] next_internal_r_quot_sel;
    reg [reg_sel_width - 1:0] next_internal_r_mod_sel;
    // reg [data_width - 1:0] next_internal_b;
    reg [data_width * 2 - 1:0] next_b_shifted;

    reg [state_width - 1:0] next_state;
    reg [pos_width - 1:0] next_pos;
    reg [data_width - 1:0] next_quotient;
    reg [data_width - 1:0] next_a_remaining;

    reg [reg_sel_width - 1:0] next_rf_wr_sel;
    reg [data_width - 1:0] next_rf_wr_data;
    reg next_rf_wr_req;
    reg next_busy;

    always @(*) begin
    // always @(state, pos, req, rf_wr_ack) begin
        // reg [2 * data_width - 1: 0] shiftedb;

        next_state = state;
        next_pos = pos;
        next_quotient = quotient;
        next_a_remaining = a_remaining;
        next_b_shifted = b_shifted;
        next_busy = 0;

        next_internal_r_quot_sel = internal_r_quot_sel;
        next_internal_r_mod_sel = internal_r_mod_sel;
        // next_internal_b = internal_b;

        next_rf_wr_req = 0;
        next_rf_wr_sel = '0;
        next_rf_wr_data = '0;

        // $display("div comb req=%0d state=%0d pos=%0d rf_wr_ack=%0d", req, state, pos, rf_wr_ack);
        // $strobe("state %0d req=%0b", state, req);
        case(state)
            IDLE: begin
                `assert_known(req);
                if (req) begin
                    // $display("div unit got req");

                    `assert_known(a);
                    `assert_known(b);
                    `assert_known(r_mod_sel);
                    `assert_known(r_quot_sel);

                    next_pos = data_width_minus_1[pos_width - 1:0];
                    next_quotient = '0;
                    next_a_remaining = a;
                    next_busy = 1;
                    next_state = CALC;

                    next_internal_r_mod_sel = r_mod_sel;
                    next_internal_r_quot_sel = r_quot_sel;
                    // next_internal_b = b;
                    next_b_shifted = { {32{1'b0}}, b} << data_width_minus_1;

                    `assert_known(b);
                    if(b == 0) begin
                        next_quotient = '1;
                        next_a_remaining = a;

                        if(r_quot_sel != 0) begin
                            next_rf_wr_req = 1;
                            next_rf_wr_sel = r_quot_sel;
                            next_rf_wr_data = next_quotient;
                            next_state = WRITING_QUOTIENT;
                            // $display("div. wrote quotient");
                        end else if(r_mod_sel != 0) begin
                            next_rf_wr_req = 1;
                            next_rf_wr_sel = r_mod_sel;
                            next_rf_wr_data = next_a_remaining;
                            next_state = WRITING_MODULUS;
                        end else begin
                            next_state = IDLE;  // this should never normally happen :), but it's a possible input
                        end
                    end
                end
            end
            CALC: begin
                next_busy = 1;

                // `assert_known(internal_b);
                // `assert_known(pos);
                // shiftedb = {{data_width{1'b0}}, internal_b} << pos;

                `assert_known(next_a_remaining);
                `assert_known(next_b_shifted);
                // `assert_known(shiftedb);
                // $display("pos %0d quot %0d a_remaining %0d shiftedb %0d", pos, next_quotient, next_a_remaining, shiftedb);
                if (b_shifted <= {{data_width{1'b0}}, next_a_remaining}) begin
                    next_a_remaining = next_a_remaining - b_shifted[data_width - 1 :0];
                    // next_quotient = next_quotient | (1 << pos);
                    next_quotient[pos] = 1;
                    // $display("   match next_quot %0d next_a %0d", next_quotient, next_a_remaining);
                end
                `assert_known(pos);
                if(pos == 0) begin
                    next_state = WRITE;
                end else begin
                    next_pos = pos - 1;
                    next_b_shifted = b_shifted >> 1;
                end
            end
            WRITE: begin
                // $display("WRITE");
                `assert_known(internal_r_quot_sel);
                `assert_known(internal_r_mod_sel);
                if(internal_r_quot_sel != 0) begin
                    next_busy = 1;
                    next_rf_wr_req = 1;
                    next_rf_wr_sel = internal_r_quot_sel;
                    next_rf_wr_data = quotient;
                    next_state = WRITING_QUOTIENT;
                    // $display("div. wrote quotient");
                end else if(internal_r_mod_sel != 0) begin
                    next_busy = 1;
                    next_rf_wr_req = 1;
                    next_rf_wr_sel = internal_r_mod_sel;
                    next_rf_wr_data = next_a_remaining;
                    next_state = WRITING_MODULUS;
                end else begin
                    next_state = IDLE;  // this should never normally happen :), but it's a possible input
                end
            end
            WRITING_QUOTIENT: begin
                // $display("WRITING_QUOTIENT");
                `assert_known(rf_wr_ack);
                if(rf_wr_ack) begin
                    // $display("div got ack, maybe write modulus");
                    `assert_known(internal_r_mod_sel);
                    if(internal_r_mod_sel != 0) begin
                        // $display("div write modulus (for next cycle)");
                        next_busy = 1;
                        next_rf_wr_req = 1;
                        next_rf_wr_sel = internal_r_mod_sel;
                        next_rf_wr_data = next_a_remaining;
                        next_state = WRITING_MODULUS;
                    end else begin
                        next_state = IDLE;
                    end
                end else begin
                //     // $display("div waiting ack");
                    next_busy = 1;
                    next_rf_wr_req = 1;
                    next_rf_wr_sel = internal_r_quot_sel;
                    next_rf_wr_data = next_quotient;
                end
            end
            WRITING_MODULUS: begin
                // $display("WRITING_MODULUS");
                `assert_known(rf_wr_ack);
                if(rf_wr_ack) begin
                    next_state = IDLE;
                end else begin
                    next_busy = 1;
                    next_rf_wr_req = 1;
                    next_rf_wr_sel = internal_r_mod_sel;
                    next_rf_wr_data = next_a_remaining;
                end
            end
            default: begin
                if(rst) $display("ERROR: got to default state state=%0d", state);
            end
        endcase
    end

    always @(posedge clk, negedge rst) begin
        // $strobe(
        //     "t=%0d int.ff state=%0d req=%0b rf_wr_req=%0d pos=%0d rst=%0b next_internal_b=%0d",
        //     $time,
        //     state, req, rf_wr_req, pos, rst, next_internal_b);
        // assert(~$isunknown(rst));
        if (~rst) begin
            state <= IDLE;
            pos <= 0;
            busy <= 0;

            quotient <= '0;
            a_remaining <= '0;
            b_shifted <= '0;
            internal_r_quot_sel <= '0;
            internal_r_mod_sel <= '0;
            // internal_b <= '0;

            rf_wr_req <= 0;
            rf_wr_sel <= 0;
            rf_wr_data <= 0;
        end else begin
            // $display("div clk");
            state <= next_state;
            pos <= next_pos;
            quotient <= next_quotient;
            a_remaining <= next_a_remaining;
            b_shifted <= next_b_shifted;

            internal_r_quot_sel <= next_internal_r_quot_sel;
            internal_r_mod_sel <= next_internal_r_mod_sel;
            // internal_b <= next_internal_b;

            rf_wr_req <= next_rf_wr_req;
            rf_wr_sel <= next_rf_wr_sel;
            rf_wr_data <= next_rf_wr_data;

            busy <= next_busy;
        end
    end
endmodule
