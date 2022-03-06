// represents processor
module proc(
    input rst, clk,
    output reg [31:0] out,
    output reg [6:0] op,
    output reg [2:0] funct,
    output reg [4:0] rd,
    output reg [4:0] rs1,
    output reg [4:0] rs2,
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
    wire [2:0] c1_funct;
    wire [4:0] c1_rd;
    wire [4:0] c1_rs1;
    wire [4:0] c1_rs2;
    wire [6:0] c1_imm1;
    wire signed [11:0] c1_store_offset;
    wire signed [11:0] c1_load_offset;

    typedef enum bit[6:0] {
        // OUT = 1,
        OUTLOC = 2,
        // LI = 3,
        // OUTR = 4,
        HALT = 5,
        STORE =    7'b0100011,
        OPIMM =    7'b0010011,
        LOAD =     7'b0000011
    } e_op;

    typedef enum bit[2:0] {
        ADDI = 3'b000
    } e_funct;

    task read_next_instr([31:0] instr_addr);
        mem_addr <= instr_addr;
        mem_rd_req <= 1;
        state <= C1;
        pc <= instr_addr;
        regs[0] <= '0;
    endtask

    task write_out([31:0] _out);
        out[31:0] <= _out;
        outen <= 1;
    endtask

    task op_imm([2:0] _funct, [4:0] _rd, [4:0] _rs1, [4:0] _rs2, [6:0] _imm1);
        case(_funct)
            ADDI: begin
                regs[_rd] <= regs[_rs1] + {_imm1, _rs2};
                read_next_instr(pc + 1);
            end
        endcase
    endtask

    task instr_c1();
        case (c1_op)
            OPIMM: begin
                op_imm(c1_funct, c1_rd, c1_rs1, c1_rs2, c1_imm1);
            end
            LOAD: begin
                // read from memory
                // lw rd, offset(rs1)
                mem_addr <= (regs[c1_rs1] + c1_load_offset) >> 2;
                mem_rd_req <= 1;
                state <= C2;
            end
            STORE: begin
                // write to memory
                // sw rs2, offset(rs1)
                if (regs[c1_rs1] + c1_store_offset == 1000) begin
                    write_out(regs[c1_rs2]);
                    read_next_instr(pc + 1);
                end else begin
                    mem_addr <= (regs[c1_rs1] + c1_store_offset) >> 2;
                    mem_wr_req <= 1;
                    mem_wr_data <= regs[c1_rs2];
                    state <= C2;
                end
            end
            /*
            OUT: begin
                write_out(c1_imm1);
                read_next_instr(pc + 1);
            end
            */
            OUTLOC: begin
                mem_addr <= {8'b0, 2'b0, c1_imm1[6:2]};
                mem_rd_req <= 1;
                state <= C2;
            end
            /*
            LI: begin
                regs[c1_rd] <= c1_imm1;
                read_next_instr(pc + 1);
            end
            OUTR: begin
                mem_wr_req <= 0;
                write_out(regs[c1_rd]);
                read_next_instr(pc + 1);
            end
            */
            HALT: begin
                halt <= 1;
            end
            default: halt <= 1;
        endcase
    endtask

    task instr_c2();
        case (op)
            LOAD: begin
                if(mem_ack) begin
                    regs[rd] <= mem_rd_data;
                    read_next_instr(pc + 1);
                end
            end
            STORE: begin
                if(mem_ack) begin
                    read_next_instr(pc + 1);
                end
            end
            OUTLOC: begin
                if(mem_ack) begin
                    write_out(mem_rd_data);
                    read_next_instr(pc + 1);
                end
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
                        funct <= mem_rd_data[14:12];
                        rd <= mem_rd_data[11:7];
                        rs1 <= mem_rd_data[19:15];
                        rs2 <= mem_rd_data[24:20];
                        // imm1 <= { {20{1'b0}}, mem_rd_data[31:25] };
                        imm1 <= mem_rd_data[31:25];
                    end
                end
                C2: begin
                    instr_c2();
                end
                default: halt <= 1;
            endcase
        end
    end
    assign c1_op = mem_rd_data[6:0];
    assign c1_rd = mem_rd_data[11:7];
    assign c1_rs1 = mem_rd_data[19:15];
    assign c1_rs2 = mem_rd_data[24:20];
    assign c1_funct = mem_rd_data[14:12];
    // assign c1_imm1 = { {20{1'b0}}, mem_rd_data[31:25] };
    assign c1_imm1 = mem_rd_data[31:25];
    assign c1_store_offset = {mem_rd_data[31:25], mem_rd_data[11:7]};
    assign c1_load_offset = mem_rd_data[31:20];
    assign x1 = regs[1];
endmodule
