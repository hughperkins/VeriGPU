module comp(input rst, clk, input [15:0]mem[32], output reg [15:0] out);
    reg [4:0] cnt;
    reg [7:0] op;
    reg [7:0] r1;

    always @(clk, cnt) begin
        op = mem[cnt][15:8];
        r1 = mem[cnt][7:0];
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 8'd0;
            cnt <= '0;
        end
        else begin
            cnt <= cnt + 1;
            //out <= mem[cnt][15:0];
            out <= r1;
        end
    end
endmodule
