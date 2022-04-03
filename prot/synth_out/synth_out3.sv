// this time, we will have a module that has state. when state is on, then it counts upwards,
// otherwise it doesnt. only the counter is visible outside the module, not hte state
// (trying to reprroduce a bug when synthesizing int_div_regfile_test.sv, and attaching to
// proc.sv)

module synth_out3(
    input clk,
    input rst,
    input req,
    output reg [3:0] cnt
);
    reg [1:0] state;
    reg [1:0] next_state;
    reg [3:0] next_cnt;

    always @(*) begin
        next_cnt = cnt;
        next_state = state;
        case(state)
            0: begin
                if(req) begin
                    next_state = 1;
                end
            end
            1: begin
                next_cnt = cnt + 1;
                if(next_cnt == 7) begin
                    next_state = 0;
                end
            end
        endcase
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            state <= 0;
            cnt <= 0;
        end else begin
            cnt <= next_cnt;
            state <= next_state;
        end
    end
endmodule
