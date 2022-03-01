module comp(input rst, clk, input [15:0]mem[32], output reg [15:0] out);
    reg [4:0] cnt;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 8'd0;
            cnt <= '0;
        end
        else begin
            cnt <= cnt + 1;
            out <= mem[cnt][15:0];
        end
    end
endmodule
