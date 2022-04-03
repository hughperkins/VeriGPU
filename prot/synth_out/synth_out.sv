module synth_out(
    input clk,
    input rst,
    input a,
    output reg out
);
    reg next_out;

    always @(*) begin
        next_out = 0;

        if(a) begin
            next_out = 1;
        end
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            out <= 0;
        end else begin
            out <= next_out;
        end
    end
endmodule
