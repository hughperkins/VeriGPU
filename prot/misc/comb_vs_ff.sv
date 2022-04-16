parameter data_width = 4;

typedef enum {
    ADD,
    SUB,
    ADDFIVE,
    MUL2,
    DIV2
} e_op;

module comb_vs_ff(
    input clk,
    input rst,
    input [3:0] op,
    output reg [data_width - 1:0] cnt
);
    // reg [data_width - 1:0] new_cnt;

    reg [data_width - 1:0] new_cnt;
    // assign new_cnt = cnt + 1;

    always @(*) begin
        // new_cnt = cnt + 1;
        case(op)
            ADD: begin
                new_cnt = cnt + 1;
            end
            SUB: begin
                new_cnt = cnt - 1;
            end
            ADDFIVE: begin
                new_cnt = cnt + 5;
            end
            MUL2: begin
                new_cnt = cnt * 2;
            end
            DIV2: begin
                new_cnt = cnt / 2;
            end
        endcase
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            cnt <= '0;
        end else begin
            cnt <= new_cnt;
            // cnt <= cnt + 1;
        end
    end
endmodule
