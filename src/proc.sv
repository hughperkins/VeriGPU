// represents processor
module proc(
    input rst, clk,
    output reg [data_width - 1:0] out,
    output reg outen,
    output reg outflen,

    output [6:0] c2_op,
    output [2:0] c2_funct,
    output [reg_sel_width - 1:0] c2_rd_sel,
    output [reg_sel_width - 1:0] c2_rs1_sel,
    output [reg_sel_width - 1:0] c2_rs2_sel,
    output [6:0] c2_imm1,

    output wire [data_width - 1:0] x1,
    output reg [addr_width - 1:0] pc,
    output reg [4:0] state,

    output reg [addr_width - 1:0] mem_addr,
    input [data_width - 1:0] mem_rd_data,
    output reg [data_width - 1:0] mem_wr_data,
    output reg mem_wr_req,
    output reg mem_rd_req,
    input mem_ack,
    input mem_busy,
    output reg halt
);
    reg [addr_width - 1:0] next_pc;
    reg [4:0] next_state;

    reg [data_width - 1:0] regs[num_regs];
    reg [instr_width - 1:0] c2_instr_next;
    reg [instr_width - 1:0] c2_instr;
    typedef enum bit[4:0] {
        C0,
        C1,
        C2
    } e_state;

    wire [6:0] c1_op;
    wire [2:0] c1_funct3;
    wire [9:0] c1_op_funct;
    wire [reg_sel_width - 1:0] c1_rd_sel;
    wire [reg_sel_width - 1:0] c1_rs1_sel;
    wire [reg_sel_width - 1:0] c1_rs2_sel;
    wire [data_width - 1:0] c1_rs1_data;
    wire [data_width - 1:0] c1_rs2_data;
    wire [6:0] c1_imm1;

    wire signed [addr_width - 1:0] c1_store_offset;
    wire signed [addr_width - 1:0] c1_load_offset;
    wire signed [data_width - 1:0] c1_i_imm;
    wire signed [addr_width - 1:0] c1_branch_offset;
    wire [instr_width - 1:0] c1_instr;

    reg [reg_sel_width - 1:0] wr_reg_sel;
    reg [data_width - 1:0] wr_reg_data;
    reg wr_reg_req;

    task read_next_instr(input [addr_width - 1:0] _next_pc);
        // assumes nothing else reading or writing to memory at same time...
        // to be be called from combinational code
        // flip-flop code will update the pc to be next_pc
        next_pc = _next_pc;
        mem_addr = next_pc;
        mem_rd_req = 1;
        next_state = C1;
    endtask

    task write_reg(input [reg_sel_width - 1:0] reg_sel, input [data_width - 1:0] reg_data);
        // to be called from combinational code
        // flip flop code will do the actual write
        // you can only call this once per clock cycle...
        // (there is no queue of these; the last register to be selected 
        // written is written; and everything else is ignored)
        wr_reg_sel = reg_sel;
        wr_reg_data = reg_data;
        wr_reg_req = 1;
    endtask

    task write_out(input [data_width - 1:0] _out);
        out = _out;
        outen = 1;
    endtask

    task write_float(input [data_width - 1:0] _out);
        out = _out;
        outflen = 1;
    endtask

    task op_imm(input [2:0] _funct, input [4:0] _rd, input [4:0] _rs1, input [data_width - 1:0] _i_imm);
        case(_funct)
            ADDI: begin
                $display("ADDI _rd=%0d regs[_rs1]=%0d _i_imm=%0d next_pc=%0d", _rd, c1_rs1_data, _i_imm, next_pc);
                write_reg(_rd, c1_rs1_data + _i_imm);
                read_next_instr(pc + 4);
            end
            default: begin
            end
        endcase
    endtask

    task op_branch(input [2:0] _funct, input [4:0] _rs1, input [4:0] _rs2, input [addr_width - 1:0] _offset);
        reg branch;
        branch = 0;
        case(_funct)
            BEQ: begin
                if (c1_rs1_data == c1_rs2_data) begin
                    branch = 1;
                end
            end
            BNE: begin
                if (c1_rs1_data != c1_rs2_data) begin
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
        wr_reg_req = 1;
        wr_reg_sel = _rd;
        case(_funct)
            ADD: wr_reg_data = c1_rs1_data + c1_rs2_data;
            // this is actually unsigned. Need to fix...
            SLT: wr_reg_data = c1_rs1_data < c1_rs2_data ? '1 : '0;
            SLTU: wr_reg_data = c1_rs1_data < c1_rs2_data ? '1 : '0;
            AND: wr_reg_data = c1_rs1_data & c1_rs2_data;
            OR: wr_reg_data = c1_rs1_data | c1_rs2_data;
            XOR: wr_reg_data = c1_rs1_data ^ c1_rs2_data;
            SLL: wr_reg_data = c1_rs1_data << c1_rs2_data[4:0];
            SRL: wr_reg_data = c1_rs1_data >> c1_rs2_data[4:0];
            SUB: wr_reg_data = c1_rs1_data - c1_rs2_data;
            // not sure what an 'arithmetic' shift is
            // need to fix...
            SRA: wr_reg_data = c1_rs1_data >> c1_rs2_data[4:0];
            // RV32M
            MUL: wr_reg_data = c1_rs1_data * c1_rs2_data;
            REM: begin end
            default: begin end
        endcase
        // $display("op regs[_rd]=%0d _rd=%0d regs[_rs1]=%0d regs[_rs2]=%0d", regs[_rd], _rd, regs[_rs1], regs[_rs2]);
        read_next_instr(pc + 4);
    endtask

    task op_lui(input [31:0] _instr, input [4:0] _rd);
        write_reg(_rd, {_instr[31:12], {12{1'b0}} });
        read_next_instr(pc + 4);
    endtask

    task op_auipc(input [31:0] _instr, input [4:0] _rd);
        write_reg(_rd, {_instr[31:12], {12{1'b0}}} + pc);
        read_next_instr(pc + 4);
    endtask

    task op_store(input [addr_width - 1:0] _addr);
        case (_addr)
            1000: begin
                // write_out(regs[c1_rs2]);
                write_out(c1_rs2_data);
                $display(" store 1000 %0d", c1_rs2_data);
                // immediately jump to next instruction, since not a real store...
                read_next_instr(pc + 4);
            end
            1004: begin
                // $display("1004: HALT");
                halt = 1;
            end
            1008: begin
                write_float(c1_rs2_data);
                read_next_instr(pc + 4);
            end
            default: begin
                // $display("default");
                // first write to memory; in C2 we will load next instruction
                mem_addr = (c1_rs1_data + c1_store_offset);
                mem_wr_req = 1;
                mem_wr_data = c1_rs2_data;
                next_state = C2;
            end
        endcase
    endtask

    task instr_c1();
        $display("instr_c1 c1_op=%0d mem_rd_data=%b", c1_op, mem_rd_data);
        // $strobe("strobe instr_c1 c1_op=%0d mem_rd_data=%b", c1_op, mem_rd_data);
        halt = 0;
        case (c1_op)
            OPIMM: begin
                $display("OPIMM");
                op_imm(c1_funct3, c1_rd_sel, c1_rs1_sel, c1_i_imm);
            end
            LOAD: begin
                $display("c1.LOAD c1_rs1=%0d regs[c1_rs1]=%0d c1_load_offset=%0d", c1_rs1_sel, regs[c1_rs1_sel], c1_load_offset);
                // read from memory
                // lw rd, offset(rs1)
                mem_addr = (c1_rs1_data + c1_load_offset);
                mem_rd_req = 1;
                next_state = C2;
            end
            STORE: begin
                $display("STORE");
                // write to memory
                // sw rs2, offset(rs1)
                op_store(c1_rs1_data + c1_store_offset);
            end
            BRANCH: begin
                // e.g. beq rs1, rs2, offset
                op_branch(c1_funct3, c1_rs1_sel, c1_rs2_sel, c1_branch_offset);
            end
            OP: begin
                op_op(c1_op_funct, c1_rd_sel, c1_rs1_sel, c1_rs2_sel);
            end
            LUI: begin
                op_lui(c1_instr, c1_rd_sel);
            end
            AUIPC: begin
                op_auipc(c1_instr, c1_rd_sel);
            end
            default: begin
                $display("default: HALT c1_op %b", c1_op);
                halt = 1;
            end
        endcase
    endtask

    task instr_c2();
        $display("C2 pc=%0d op=%0d %0b", pc, c2_op, c2_op);
        case (c2_op)
            LOAD: begin
                // $display("C2.load mem_ack=%0b", mem_ack);
                if(mem_ack) begin
                    $display("C2.load next c2_rd_sel=%0d mem_rd_data=%0d", c2_rd_sel, mem_rd_data);
                    write_reg(c2_rd_sel, mem_rd_data);
                    read_next_instr(pc + 4);
                end
            end
            STORE: begin
                // $display("C2.store mem_ack %0b", mem_ack);
                if(mem_ack) begin
                    read_next_instr(pc + 4);
                end
            end
            default: begin
            end
        endcase
    endtask

    always @(*) begin
    // always_comb begin
        halt = 0;
        outen = 0;
        outflen = 0;
        mem_addr = '0;
        mem_rd_req = 0;
        mem_wr_req = 0;
        mem_wr_data = '0;
        next_pc = pc;
        next_state = state;

        c2_instr_next = c2_instr;

        wr_reg_sel = '0;
        wr_reg_data = '0;
        wr_reg_req = 0;

        if(~rst) begin
            $display("comb t=%0d state=%0d pc=%0d c1_op=%0d mem_wr_req=%0b mem_rd_req=%0b mem_ack=%0b regs[1]=%0d", $time, state, pc, c1_op, mem_wr_req, mem_rd_req, mem_ack, regs[1]);
        end
        case(state)
            C0: begin
                // if(~rst) begin
                //     $display("comb C0");
                // end
                read_next_instr(pc);
            end
            C1: begin
                // $display("comb C1");
                mem_rd_req = 0;
                if(mem_ack) begin
                    // $display("in mem_ack");
                    instr_c1();
                    c2_instr_next = mem_rd_data;
                end
            end
            C2: begin
                // $display("Comb.C2");
                instr_c2();
            end
            default: begin
                // $display("Comb.default");
                halt = 1;
            end
        endcase
        // if(~rst) begin
        //     $display("comb end mem_rd_req=%0b",mem_rd_req);
        // end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pc <= 0;
            state <= C0;
            regs[0] <= '0;
        end else begin
            $display(
                "ff mem_addr %0d mem_wr_data %0d mem_rd_data %0d mem_wr_req %b mem_rd_req  %b mem_ack %b mem_busy %b",
                mem_addr,     mem_wr_data,    mem_rd_data,    mem_wr_req,   mem_rd_req,    mem_ack,   mem_busy);
            $display("ff tick t=%0d clk=%0b next_pc=%0d next_state=%0d", $time, clk, next_pc, next_state);
            pc <= next_pc;
            state <= next_state;
            c2_instr <= c2_instr_next;
            if (wr_reg_req) begin
                regs[wr_reg_sel] <= wr_reg_data;
            end
        end
    end

    assign c1_rs1_data = regs[c1_rs1_sel];
    assign c1_rs2_data = regs[c1_rs2_sel];
    assign c1_op = mem_rd_data[6:0];
    assign c1_rd_sel = mem_rd_data[11:7];
    assign c1_rs1_sel = mem_rd_data[19:15];
    assign c1_rs2_sel = mem_rd_data[24:20];
    assign c1_funct3 = mem_rd_data[14:12];
    assign c1_imm1 = mem_rd_data[31:25];
    assign c1_instr = mem_rd_data;
    assign c1_store_offset = {{20{mem_rd_data[31]}}, mem_rd_data[31:25], mem_rd_data[11:7]};
    assign c1_load_offset = {{20{mem_rd_data[31]}}, mem_rd_data[31:20]};
    assign c1_i_imm = {{20{mem_rd_data[31]}}, mem_rd_data[31:20]};
    assign c1_branch_offset = {{20{mem_rd_data[31]}}, mem_rd_data[31], mem_rd_data[7], mem_rd_data[30:25], mem_rd_data[11:8]};
    assign c1_op_funct = {mem_rd_data[31:25], mem_rd_data[14:12]};
    assign x1 = regs[1];

    assign c2_op = c2_instr[6:0];
    assign c2_rd_sel = c2_instr[11:7];
endmodule
