// test using timing.py for a module with posedge always, and a case, with
// several maths functions
module posedge_case(input clk, input rst, input op, output reg [31:0] q);
    always @(posedge clk, posedge rst) begin
        if(rst) begin
            q <= '0;
        end else begin
            case (op)
                0: begin
                    q <= q + 1;
                end
                1: begin
                    q <= q * 3;
                end
            endcase
        end
    end
endmodule
