module comp(input rst, clk, input [15:0]mem[256], output reg [15:0] out, output reg [7:0] op, output reg [7:0] p1);
    reg [4:0] cnt;
    //reg [7:0] op;
    //reg [7:0] p1;
    reg [7:0] r1;

    always @(clk, cnt) begin
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 8'd0;
            cnt <= '0;
        end
        else begin
            op = mem[cnt][15:8];
            p1 = mem[cnt][7:0];
            out = '0;
            case (op)
               8'h01: out = p1;
               8'h02: begin
                  out = mem[p1 >> 1];
               end
            endcase
            cnt <= cnt + 1;
            //out <= mem[cnt][15:0];
            //out <= r1;
        end
    end
endmodule
