//
// test how to do aserts in tet scripts, etc
module test_test_timings(
    input clk,
    input rst,
    input req,
    input clr,
    output reg out
);
    reg n_out;

    always @(*) begin
        n_out = 0;
        if (req) begin
            n_out = 1;
        end
        if(clr) begin
            n_out = 0;
        end
    end

    always @(posedge clk, posedge rst) begin
        if(rst) begin
            out <= 0;
        end else begin
            out <= n_out;
        end
    end
endmodule
