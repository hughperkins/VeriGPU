module comp(
    input rst, clk,
    input [15:0]mem[256],
    output reg [15:0] out,
    output reg [3:0] op,
    output reg [3:0] reg_select,
    output reg [7:0] p1,
    output reg [7:0] x1
   // output reg [7:0] regs[32]
);
    reg [4:0] cnt;
    //reg [7:0] op;
    //reg [7:0] p1;
    //reg [7:0] r1;
    reg [7:0] regs[32];

    always @(clk, cnt) begin
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 8'd0;
            cnt <= '0;
        end
        else begin
            op = mem[cnt][11:8];
            reg_select = mem[cnt][15:12];
            p1 = mem[cnt][7:0];
            out = '0;
            case (op)
               4'h1: out = p1;
               4'h2: begin
                  out = mem[p1 >> 1];
               end
               4'h3: begin
                  regs[reg_select] = p1;
               end
               4'h4: begin
                   out = regs[reg_select];
               end
            endcase
            //out <= mem[cnt][15:12];
            //regs[1] = 8'hab;
            x1 = regs[1];
            cnt <= cnt + 1;
            //out <= mem[cnt][15:0];
            //out <= r1;
        end
    end
endmodule
