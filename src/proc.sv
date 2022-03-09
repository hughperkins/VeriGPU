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
    wire [2:0] c1_funct3;
    wire [9:0] c1_op_funct;
    wire [4:0] c1_rd;
    wire [4:0] c1_rs1;
    wire [4:0] c1_rs2;
    wire [6:0] c1_imm1;

    wire signed [31:0] c1_store_offset;
    wire signed [31:0] c1_load_offset;
    wire signed [31:0] c1_i_imm;
    wire signed [31:0] c1_branch_offset;

    typedef enum bit[6:0] {
        STORE =    7'b0100011,
        OPIMM =    7'b0010011,
        LOAD =     7'b0000011,
        BRANCH =   7'b1100011,
        OP =       7'b0110011,
        LUI =      7'b0110111,
        MADD =     7'b1000011,
        LOADFP =   7'b0000111,
        STOREFP =  7'b0100111,
        MSUB =     7'b1000111,
        JALR =     7'b1100111,
        NMSUB =    7'b1001011,
        NMADD =    7'b1001111,
        OPFP =     7'b1010011,
        AUIPC =    7'b0010111,
        OP32 =     7'b0111011,
        OPIMM32 =  7'b0011011
    } e_op;

    typedef enum bit[2:0] {
        ADDI = 3'b000
    } e_funct;

    typedef enum bit[2:0] {
        BEQ = 3'b000,
        BNE = 3'b001,
        BLT = 3'b100,
        BGE = 3'b101,
        BLTU = 3'b110,
        BGEU = 3'b111
    } e_funct_branch;

    typedef enum bit[9:0] {
        ADD =  10'b0000000000,
        SLT =  10'b0000000010,
        SLTU = 10'b0000000011,
        AND =  10'b0000000111,
        OR =   10'b0000000110,
        XOR =  10'b0000000100,
        SLL =  10'b0000000001,
        SRL =  10'b0000000101,
        SUB =  10'b0100000000,
        SRA =  10'b0100000101,

        // RV32M
        MUL =    10'b0000001000,
        MULH =   10'b0000001001,
        MULHSU = 10'b0000001010,
        MULHU =  10'b0000001011,
        DIV =    10'b0000001100,
        DIVU =   10'b0000001101,
        REM =    10'b0000001110,
        REMU =   10'b0000001111
    } e_funct_op;

    task read_next_instr(input [31:0] instr_addr);
        mem_addr <= instr_addr;
        mem_rd_req <= 1;
        state <= C1;
        pc <= instr_addr;
        regs[0] <= '0;
    endtask

    task write_out(input [31:0] _out);
        out[31:0] <= _out;
        outen <= 1;
    endtask

    task op_imm(input [2:0] _funct, input [4:0] _rd, input [4:0] _rs1, input [31:0] _i_imm);
        case(_funct)
            ADDI: begin
                regs[_rd] <= regs[_rs1] + _i_imm;
                read_next_instr(pc + 4);
            end
            default: begin
            end
        endcase
    endtask

    task op_branch(input [2:0] _funct, input [4:0] _rs1, input [4:0] _rs2, input [31:0] _offset);
        reg branch;
        branch = 0;
        case(_funct)
            BEQ: begin
                if (regs[_rs1] == regs[_rs2]) begin
                    branch = 1;
                end
            end
            BNE: begin
                if (regs[_rs1] != regs[_rs2]) begin
                    branch = 1;
                end
            end
            default: begin
            end
        endcase

        if (branch) begin
            read_next_instr(pc + {_offset[30:0], 1'b0});
        end else begin
            read_next_instr(pc + 4);
        end
    endtask

    task op_op(input [9:0] _funct, input [4:0] _rd, input [4:0] _rs1, input [4:0] _rs2);
        case(_funct)
            ADD: begin
                regs[_rd] <= regs[_rs1] + regs[_rs2];
            end
            SLT: begin
                // this is actually unsigned. Need to fix...
                regs[_rd] <= regs[_rs1] < regs[_rs2] ? '1 : '0;
            end
            SLTU: begin
                regs[_rd] <= regs[_rs1] < regs[_rs2] ? '1 : '0;
            end
            AND: begin
                regs[_rd] <= regs[_rs1] & regs[_rs2];
            end
            OR: begin
                regs[_rd] <= regs[_rs1] | regs[_rs2];
            end
            XOR: begin
                regs[_rd] <= regs[_rs1] ^ regs[_rs2];
            end
            SLL: begin
                regs[_rd] <= regs[_rs1] << regs[_rs2][4:0];
            end
            SRL: begin
                regs[_rd] <= regs[_rs1] >> regs[_rs2][4:0];
            end
            SUB: begin
                regs[_rd] <= regs[_rs1] - regs[_rs2];
            end
            SRA: begin
                // not sure what an 'arithmetic' shift is
                // need to fix...
                regs[_rd] <= regs[_rs1] >> regs[_rs2][4:0];
            end

            // RV32M
            MUL: begin
                regs[_rd] <= regs[_rs1] * regs[_rs2];
            end
            REM: begin
            end

            default: begin
            end
        endcase
        read_next_instr(pc + 4);
    endtask

    task instr_c1();
        case (c1_op)
            OPIMM: begin
                op_imm(c1_funct3, c1_rd, c1_rs1, c1_i_imm);
            end
            LOAD: begin
                // read from memory
                // lw rd, offset(rs1)
                mem_addr <= (regs[c1_rs1] + c1_load_offset);
                mem_rd_req <= 1;
                state <= C2;
            end
            STORE: begin
                // write to memory
                // sw rs2, offset(rs1)
                if (regs[c1_rs1] + c1_store_offset == 1000) begin
                    write_out(regs[c1_rs2]);
                    read_next_instr(pc + 4);
                end else if(regs[c1_rs1] + c1_store_offset == 1004) begin
                    halt <= 1;
                end else begin
                    mem_addr <= (regs[c1_rs1] + c1_store_offset);
                    mem_wr_req <= 1;
                    mem_wr_data <= regs[c1_rs2];
                    state <= C2;
                end
            end
            BRANCH: begin
                // e.g. beq rs1, rs2, offset
                op_branch(c1_funct3, c1_rs1, c1_rs2, c1_branch_offset);
            end
            OP: begin
                op_op(c1_op_funct, c1_rd, c1_rs1, c1_rs2);
            end
            default: begin
                halt <= 1;
            end
        endcase
    endtask

    task instr_c2();
        case (op)
            LOAD: begin
                if(mem_ack) begin
                    regs[rd] <= mem_rd_data;
                    read_next_instr(pc + 4);
                end
            end
            STORE: begin
                if(mem_ack) begin
                    read_next_instr(pc + 4);
                end
            end
            default: begin
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
    assign c1_funct3 = mem_rd_data[14:12];
    assign c1_imm1 = mem_rd_data[31:25];
    assign c1_store_offset = {{20{mem_rd_data[31]}}, mem_rd_data[31:25], mem_rd_data[11:7]};
    assign c1_load_offset = {{20{mem_rd_data[31]}}, mem_rd_data[31:20]};
    assign c1_i_imm = {{20{mem_rd_data[31]}}, mem_rd_data[31:20]};
    assign c1_branch_offset = {{20{mem_rd_data[31]}}, mem_rd_data[31], mem_rd_data[7], mem_rd_data[30:25], mem_rd_data[11:8]};
    assign c1_op_funct = {mem_rd_data[31:25], mem_rd_data[14:12]};
    assign x1 = regs[1];
endmodule
