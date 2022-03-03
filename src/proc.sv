// represents processor
module proc(
    input rst, clk,
    output reg [15:0] out,
    output [3:0] op,
    output [3:0] reg_select,
    output [7:0] p1,
    output [7:0] x1,
    output reg [15:0] pc,

    output reg mem_we,
    output [15:0] mem_read_addr,
    output reg [15:0] mem_write_addr,
    input [15:0] mem_read_data,
    output reg [15:0] mem_write_data
);
    //reg [7:0] cnt;
    // reg [15:0] pc;
    //reg [7:0] op;
    //reg [7:0] p1;
    //reg [7:0] r1;
    reg [7:0] regs[16];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= '0;
            // cnt <= '0;
            pc <= '0;
        end
        else begin
            mem_we <= 1;
            // mem_read_addr <= pc;
            // TODO p1 = mem[cnt][7:0];
            out <= '0;
            case (op)
               4'h1: out[7:0] <= p1;
               4'h2: begin
                   // TODO out <= mem[p1 >> 1];
               end
               4'h3: begin
                  regs[reg_select] <= p1;
               end
               4'h4: begin
                   out[7:0] <= regs[reg_select];
               end
               default: out <= '0;
            endcase
            //out <= mem[cnt][15:12];
            //regs[1] = 8'hab;
            // x1 = regs[1];
            //cnt <= cnt + 1;
            pc <= pc + 1;
            //out <= mem[cnt][15:0];
            //out <= r1;
        end
    end
    assign mem_read_addr = pc;
    assign op = mem_read_data[11:8];
    assign reg_select = mem_read_data[15:12];
    assign p1 = mem_read_data[7:0];
    assign x1 = regs[1];
endmodule
