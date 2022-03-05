// represents processor
module proc(
    input rst, clk,
    output reg [31:0] out,
    output reg [6:0] op,
    output reg [4:0] rd,
    output reg [6:0] imm1,
    output [31:0] x1,
    output reg [31:0] pc,
    output reg [4:0] state,
    output reg outen,

    output reg [31:0] mem_addr,
    input [31:0] mem_rd_data,
    output reg [31:0] mem_wr_data,
    output reg mem_wr_req,
    output reg mem_rd_req,
    input mem_ack,
    input mem_busy,
    output reg halt
);
    reg [31:0] regs[32];
    reg [31:0] instruction;
    typedef enum bit[4:0] {
        C1,
        C2
    } e_state;

    wire [6:0] c1_op;
    wire [4:0] c1_rd;
    wire [31:0] c1_imm1;

    typedef enum bit[3:0] {
        OUT = 1,
        OUTLOC = 2,
        LI = 3,
        OUTR = 4,
        HALT = 5
    } e_op;

    task read_next_instr([31:0] instr_addr);
        mem_addr <= instr_addr;
        mem_rd_req <= 1;
        state <= C1;
        pc <= instr_addr;
    endtask

    task write_out([31:0] _out);
        out[31:0] <= _out;
        outen <= 1;
    endtask

    task instr_c1();
        case (c1_op)
            OUT: begin
                write_out(c1_imm1);
                read_next_instr(pc + 1);
            end
            OUTLOC: begin
                mem_addr <= {8'b0, 2'b0, c1_imm1[6:2]};
                mem_rd_req <= 1;
                state <= C2;
            end
            LI: begin
                regs[c1_rd] <= c1_imm1;
                read_next_instr(pc + 1);
            end
            OUTR: begin
                mem_wr_req <= 0;
                write_out(regs[c1_rd]);
                read_next_instr(pc + 1);
           end
           HALT: begin
               halt <= 1;
           end
           default: halt <= 1;
        endcase
    endtask

    task instr_c2();
        case (op)
            OUTLOC: begin
                write_out(mem_rd_data);
                read_next_instr(pc + 1);
            end
        endcase
    endtask

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            halt <= 0;
            outen <= 0;
            read_next_instr(0);
        end
        else begin
            mem_rd_req <= 0;
            mem_wr_req <= 0;
            outen <= 0;
            case(state)
                C1: begin
                    mem_rd_req <= 0;
                    if(mem_ack) begin
                        instr_c1();
                        instruction <= mem_rd_data;
                        op <= mem_rd_data[6:0];
                        rd <= mem_rd_data[11:7];
                        imm1 <= { {20{1'b0}}, mem_rd_data[31:25] };
                    end
                end
                C2: begin
                    if(mem_ack) begin
                        instr_c2();
                    end
                end
                default: halt <= 1;
            endcase
        end
    end
    assign c1_op = mem_rd_data[6:0];
    assign c1_rd = mem_rd_data[11:7];
    assign c1_imm1 = { {20{1'b0}}, mem_rd_data[31:25] };
    assign x1 = regs[1];
endmodule
