/*
 epresents processor, but not including any memory elements, such as mem and reg_file, which are slow to synthesize

As of mar 28 2022:
Max propagation delay: 82.8 nand units
Area:                  30956.0 nand units
 */
module proc(
    input rst, clk,
    output reg [data_width - 1:0] out,
    output reg outen,
    output reg outflen,

    output reg [data_width - 1:0] x1,
    output reg [addr_width - 1:0] pc,
    output reg [4:0] state,

    output reg [addr_width - 1:0] mem_addr,
    input [data_width - 1:0]      mem_rd_data,
    output reg [data_width - 1:0] mem_wr_data,
    output reg                    mem_wr_req,
    output reg                    mem_rd_req,
    input                         mem_ack,
    input                         mem_busy,

    output reg halt
);
    reg [addr_width - 1:0] next_pc;
    reg [4:0]              next_state;

    reg [data_width - 1:0]  regs[num_regs];

    reg [instr_width - 1:0] c2_instr_next;
    reg [instr_width - 1:0] c2_instr;
    typedef enum bit[4:0] {
        C0,
        C1,
        C2
    } e_state;

    reg [6:0]                 c1_op;
    reg [2:0]                 c1_funct3;
    reg [9:0]                 c1_op_funct;
    reg [reg_sel_width - 1:0] c1_rd_sel;
    reg [reg_sel_width - 1:0] c1_rs1_sel;
    reg [reg_sel_width - 1:0] c1_rs2_sel;
    reg [data_width - 1:0]    c1_rs1_data;
    reg [data_width - 1:0]    c1_rs2_data;
    reg [6:0] c1_imm1;

    reg signed [addr_width - 1:0] c1_store_offset;
    reg signed [addr_width - 1:0] c1_load_offset;
    reg signed [data_width - 1:0] c1_i_imm;
    reg signed [addr_width - 1:0] c1_branch_offset;
    reg [instr_width - 1:0]       c1_instr;

    reg [6:0] c2_op;
    reg [9:0] c2_op_funct;
    reg [reg_sel_width - 1:0] c2_rd_sel;
    reg [reg_sel_width - 1:0] c2_rs1_sel;
    reg [reg_sel_width - 1:0] c2_rs2_sel;
    reg [6:0] c2_imm1;

    reg [reg_sel_width - 1:0] wr_reg_sel;
    reg [data_width - 1:0]    wr_reg_data;
    reg wr_reg_req;

    reg mul_req;
    reg mul_ack;
    reg [data_width - 1:0] mul_a;  // TODO: maybe merge all a and b, across all operations?
    reg [data_width - 1:0] mul_b;
    reg [data_width - 1:0] mul_out;

    reg n_mul_req;
    reg [float_width - 1:0] n_mul_a;
    reg [float_width - 1:0] n_mul_b;

    mul_pipeline_32bit mul_pipeline_32bit_(
        .clk(clk),
        .rst(rst),
        .req(mul_req),
        .a(mul_a),
        .b(mul_b),
        .out(mul_out),
        .ack(mul_ack)
    );

    reg                       n_div_req;
    reg [reg_sel_width - 1:0] n_div_r_quot_sel;
    reg [reg_sel_width - 1:0] n_div_r_mod_sel;
    reg [data_width - 1:0]    n_div_rs1_data;
    reg [data_width - 1:0]    n_div_rs2_data;

    reg                       div_req;
    reg [reg_sel_width - 1:0] div_r_quot_sel;
    reg [reg_sel_width - 1:0] div_r_mod_sel;
    reg [data_width - 1:0]    div_rs1_data;
    reg [data_width - 1:0]    div_rs2_data;

    reg                       div_wr_reg_ack;

    reg                       div_busy;
    reg [reg_sel_width - 1:0] div_wr_reg_sel;
    reg [data_width - 1:0]    div_wr_reg_data;
    reg                       div_wr_reg_req;

    int_div_regfile int_div_regfile_(
        .clk(clk),
        .rst(rst),
        .req(div_req),
        .busy(div_busy),
        .r_quot_sel(div_r_quot_sel),
        .r_mod_sel(div_r_mod_sel),
        .a(div_rs1_data),
        .b(div_rs2_data),
        .rf_wr_sel(div_wr_reg_sel),
        .rf_wr_data(div_wr_reg_data),
        .rf_wr_req(div_wr_reg_req),
        .rf_wr_ack(div_wr_reg_ack)
    );

    reg fadd_req;
    reg fadd_ack;
    reg [float_width - 1:0] fadd_a;
    reg [float_width - 1:0] fadd_b;
    reg [float_width - 1:0] fadd_out;

    reg n_fadd_req;
    reg [float_width - 1:0] n_fadd_a;
    reg [float_width - 1:0] n_fadd_b;

    float_add_pipeline float_add_pipeline_(
        .clk(clk),
        .rst(rst),
        .req(fadd_req),
        .ack(fadd_ack),
        .a(fadd_a),
        .b(fadd_b),
        .out(fadd_out)
    );

    reg fmul_req;
    reg fmul_ack;
    reg [float_width - 1:0] fmul_a;
    reg [float_width - 1:0] fmul_b;
    reg [float_width - 1:0] fmul_out;

    reg n_fmul_req;
    reg [float_width - 1:0] n_fmul_a;
    reg [float_width - 1:0] n_fmul_b;

    float_mul_pipeline float_mul_pipeline_(
        .clk(clk),
        .rst(rst),
        .req(fmul_req),
        .ack(fmul_ack),
        .a(fmul_a),
        .b(fmul_b),
        .out(fmul_out)
    );

    task read_mem(input [addr_width - 1:0] addr);
        // combinatorial task
        // sends off read mem request
        // you still have to wait for mem_ack, and then read
        // mem_rd_data
        // can only call this once per tick, later calls in same
        // tick will simply overwrite earlier calls...
        mem_addr = addr;
        mem_rd_req = 1;
    endtask

    task write_mem(input [addr_width - 1:0] addr, input [data_width - 1:0] data);
        // combinatorial task
        // sends off write mem request
        // you still have to wait for mem_ack
        // can only call this once per tick, later calls in same
        // tick will simply overwrite earlier calls...
        // also, read_mem and write_mem use the same address port, so you can only
        // call one or the other in a single tick too
        mem_addr = addr;
        mem_wr_data = data;
        mem_wr_req = 1;
    endtask

    task read_next_instr(input [addr_width - 1:0] _next_pc);
        // assumes nothing else reading or writing to memory at same time...
        // to be be called from combinational code
        // flip-flop code will update the pc to be next_pc
        next_pc = _next_pc;
        read_mem(next_pc);
        next_state = C1;
    endtask

    task write_reg(input [reg_sel_width - 1:0] reg_sel, input [data_width - 1:0] reg_data);
        // to be called from combinational code
        // flip flop code will do the actual write
        // you can only call this once per clock cycle...
        // (there is no queue of these; the last call overwrites any earlier calls)
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
        `assert_known(_funct);
        case(_funct)
            ADDI: begin
                $display("%0d ADDI x%0d <= %0d + %0d", pc, _rd, c1_rs1_data, _i_imm);
                write_reg(_rd, c1_rs1_data + _i_imm);
            end
            SLTI: begin
                // FIXME: this should not be the same as SLTIU
                write_reg(_rd, { {31{1'b0}}, (c1_rs1_data < _i_imm)});
            end
            SLTIU: begin
                $display("%0d SLTIU x%0d <= %0d < %0d", pc, _rd, c1_rs1_data, c1_rs2_data);
                write_reg(_rd, { {31{1'b0}}, (c1_rs1_data < _i_imm)});
            end
            XORI: begin
                write_reg(_rd, c1_rs1_data ^ _i_imm);
            end
            ORI: begin
                write_reg(_rd, c1_rs1_data | _i_imm);
            end
            ANDI: begin
                write_reg(_rd, c1_rs1_data & _i_imm);
            end
            SLLI: begin
                $display("%0d SLLI x%0d <= %0d << %0d", pc, _rd, c1_rs1_data, c1_rs2_sel);
                write_reg(_rd, (c1_rs1_data << c1_rs2_sel));
            end
            SRLI: begin
                $display("%0d SRLI x%0d <= %0d >> %0d", pc, _rd, c1_rs1_data, c1_rs2_sel);
                write_reg(_rd, (c1_rs1_data >> c1_rs2_sel));
            end
            // SRAI: begin FIXME: implement
            // end
            default: begin
                $display("op immm unhandled funct %0d", _funct);
                halt = 1;
            end
        endcase
        read_next_instr(pc + 4);
    endtask

    task op_branch(input [2:0] _funct, input [4:0] _rs1, input [4:0] _rs2, input [addr_width - 1:0] _offset);
        reg branch;
        reg [addr_width - 1:0] branch_dest;

        branch = 0;
        branch_dest = pc + {_offset[30:0], 1'b0};

        `assert_known(_funct);
        case(_funct)
            BEQ: begin
                $display("%0d BEQ %0d == %0d => %0d", pc, c1_rs1_data, c1_rs2_data, branch_dest);
                if (c1_rs1_data == c1_rs2_data) branch = 1;
            end
            BNE: begin
                $display("%0d BNE %0d != %0d => %0d", pc, c1_rs1_data, c1_rs2_data, branch_dest);
                if (c1_rs1_data != c1_rs2_data) branch = 1;
            end
            BGE: begin
                $display("%0d BGE %0d >= %0d => %0d", pc, c1_rs1_data, c1_rs2_data, branch_dest);
                if (c1_rs1_data >= c1_rs2_data) branch = 1;
            end
            BLT: begin  //fix me: BLT and BLTU should not be the same... (ditto for BGE and BGEU)
                $display("%0d BLT %0d < %0d => %0d", pc, c1_rs1_data, c1_rs2_data, branch_dest);
                if (c1_rs1_data < c1_rs2_data) branch = 1;
            end
            BGEU: begin
                $display("%0d BGEU %0d >= %0d => %0d", pc, c1_rs1_data, c1_rs2_data, branch_dest);
                if (c1_rs1_data >= c1_rs2_data) branch = 1;
            end
            BLTU: begin
                $display("%0d BLTU %0d < %0d => %0d", pc, c1_rs1_data, c1_rs2_data, branch_dest);
                if (c1_rs1_data < c1_rs2_data) branch = 1;
            end
            default: begin end
        endcase

        `assert_known(branch);
        if (branch) begin
            read_next_instr(branch_dest);
        end else begin
            read_next_instr(pc + 4);
        end
    endtask

    task op_fp(
        input [instr_width - 1:0] _instr,
        input [4:0] _rd_sel,
        input [data_width - 1:0] _rs1_data,
        input [data_width - 1:0] _rs2_data
    );
        reg [4:0] funct5;
        funct5 = _instr[31:27];
        `assert_known(funct5);
        case (funct5)
            FADD: begin
                $display("FADD.C1 x%0d <=", _rd_sel);
                n_fadd_req = 1;
                n_fadd_a = _rs1_data;
                n_fadd_b = _rs2_data;
                next_state = C2;
            end
            FMUL: begin
                $display("FMUL.C1 x%0d <=", _rd_sel);
                n_fmul_req = 1;
                n_fmul_a = _rs1_data;
                n_fmul_b = _rs2_data;
                next_state = C2;
            end
            default: begin
                $display("op_fp case funct5 default shoult not be hit");
                halt = 1;
            end
        endcase
    endtask

    task op_fp_c2(
        input [instr_width - 1:0] _instr,
        input [4:0] _rd_sel
    );
        reg [4:0] funct5;
        funct5 = _instr[31:27];
        `assert_known(funct5);
        // $display("op_fp_c2");
        case (funct5)
            FADD: begin
                // $display("FADD.C2");
                `assert_known(fadd_ack);
                if(fadd_ack) begin
                    $display("FADD.C2 x%0d <= %b", _rd_sel, fadd_out);
                    wr_reg_data = fadd_out;
                    wr_reg_sel = _rd_sel;
                    wr_reg_req = 1;
                    read_next_instr(pc + 4);
                end
            end
            FMUL: begin
                // $display("FMUL.C2");
                `assert_known(fmul_ack);
                if(fmul_ack) begin
                    $display("FMUL.C2 x%0d <= %b", _rd_sel, fmul_out);
                    wr_reg_data = fmul_out;
                    wr_reg_sel = _rd_sel;
                    wr_reg_req = 1;
                    read_next_instr(pc + 4);
                end
            end
            default: begin
                $display("op_fp_c2 case funct5 default shoult not be hit");
                halt = 1;
            end
        endcase
    endtask

    task op_op(input [9:0] _funct, input [4:0] _rd_sel, input [4:0] _rs1_sel, input [4:0] _rs2_sel);
        reg skip_advance_pc;

        wr_reg_req = 1;
        wr_reg_sel = _rd_sel;
        // $display("op_op.c1 op_funct=%0d", _funct);
        skip_advance_pc = 0;
        `assert_known(_funct);
        case(_funct)
            ADD: begin
                $display("%0d ADD x%0d <= %0d + %0d", pc, _rd_sel, c1_rs1_data, c1_rs2_data);
                chunked_add_task(
                    c1_rs1_data,
                    c1_rs2_data,
                    wr_reg_data
                );
            end
            // fixme: this should be signed
            SLT: wr_reg_data = c1_rs1_data < c1_rs2_data ? 1 : 0;
            SLTU: begin
                $display("%0d SLTU x%0d <= %0d < %0d", pc, _rd_sel, c1_rs1_data, c1_rs2_data);
                wr_reg_data = c1_rs1_data < c1_rs2_data ? 1 : 0;
            end
            AND: begin
                $display("%0d AND x%0d <= %0d & %0d", pc, _rd_sel, c1_rs1_data, c1_rs2_data);
                wr_reg_data = c1_rs1_data & c1_rs2_data;
            end
            OR: wr_reg_data = c1_rs1_data | c1_rs2_data;
            XOR: wr_reg_data = c1_rs1_data ^ c1_rs2_data;
            SLL: wr_reg_data = c1_rs1_data << c1_rs2_data[4:0];
            SRL: wr_reg_data = c1_rs1_data >> c1_rs2_data[4:0];
            SUB: begin
                chunked_sub_task(
                    c1_rs1_data,
                    c1_rs2_data,
                    wr_reg_data
                );
            end
            // fixme: SRA is currently unsigned
            SRA: wr_reg_data = c1_rs1_data >> c1_rs2_data[4:0];
            // RV32M
            MUL: begin
                $display("%0d MUL.c1 %0d * %0d => x%0d", pc, c1_rs1_data, c1_rs2_data, _rd_sel);
                n_mul_req = 1;
                n_mul_a = c1_rs1_data;
                n_mul_b = c1_rs2_data;
                next_state = C2;
                skip_advance_pc = 1;
            end
            DIVU: begin
                $display("%0d DIVU.c1 %0d / %0d => x%0d", pc, c1_rs1_data, c1_rs2_data, _rd_sel);
                `assert_known(div_busy);
                if(div_busy == 0) begin
                    // $display("sending req to div unit a=%0d b=%0d quot_sel=%0d", c1_rs1_data, c1_rs2_data, _rd_sel);
                    n_div_req = 1;
                    n_div_r_quot_sel = _rd_sel;
                    n_div_r_mod_sel = '0;
                    n_div_rs1_data = c1_rs1_data;
                    n_div_rs2_data = c1_rs2_data;
                    // since we havent implemented any kind of instruction parallelism (i.e. wiatin for egister to 
                    // be vailable...), so we need to move to next state, and wait for div to finish first
                    // we can improve this later
                    next_state = C2;
                    skip_advance_pc = 1;
                end else begin
                    // $display("waiting for div unit to be free");
                    skip_advance_pc = 1;
                end
            end
            REMU: begin
                $display("%0d REMU.c1 x%0d <= %0d / %0d", pc, _rd_sel, c1_rs1_data, c1_rs2_data);
                `assert_known(div_busy);
                if(~div_busy) begin
                    // $display("sending req to div unit a=%0d b=%0d mod_sel=%0d", c1_rs1_data, c1_rs2_data, _rd_sel);
                    n_div_req = 1;
                    n_div_r_quot_sel = '0;
                    n_div_r_mod_sel = _rd_sel;
                    n_div_rs1_data = c1_rs1_data;
                    n_div_rs2_data = c1_rs2_data;
                    // since we havent implemented any kind of instruction parallelism (i.e. wiatin for egister to 
                    // be vailable...), so we need to move to next state, and wait for div to finish first
                    // we can improve this later
                    next_state = C2;
                    skip_advance_pc = 1;
                end else begin
                    // wait for not busy I suppose...
                    // $display("waiting for div unit to be free");
                    skip_advance_pc = 1;
                end
            end
            default: begin
                $display("ERROR: unknown _funct=%b", _funct);
            end
        endcase
        // $display("op regs[_rd]=%0d _rd=%0d regs[_rs1]=%0d regs[_rs2]=%0d", regs[_rd], _rd, regs[_rs1], regs[_rs2]);
        `assert_known(skip_advance_pc);
        if(~skip_advance_pc) begin
            read_next_instr(pc + 4);
        end
    endtask

    task op_lui(input [31:0] _instr, input [4:0] _rd);
        write_reg(_rd, {_instr[31:12], {12{1'b0}} });
        read_next_instr(pc + 4);
    endtask

    task op_auipc(input [31:0] _instr, input [4:0] _rd);
        write_reg(_rd, {_instr[31:12], {12{1'b0}}} + pc);
        read_next_instr(pc + 4);
    endtask

    task op_jal(input [31:0] _instr, input [4:0] _rd_sel);
        reg signed [31:0] imm;
        reg [31:0] next_pc;

        $display("_instr %b", _instr);
        $display("_instr[31] %b", _instr[31]);
        imm = { {11{_instr[31]}}, _instr[31], _instr[19:12], _instr[20], _instr[30:21], 1'b0 };
        $display("imm %b %0d", imm, imm);
        write_reg(_rd_sel, pc + 4);
        next_pc = imm + pc;
        $display("pc %0d next_pc %0d", pc, next_pc);
        $display("JAL storing %0d in x%0d", pc + 4, _rd_sel);
        read_next_instr(next_pc);
    endtask

    task op_jalr(input [31:0] _instr, input [4:0] _rd_sel, input [data_width - 1:0] _rs1_data);
        reg signed [31:0] imm;
        reg [31:0] next_pc;

        $display("_instr %b", _instr);
        $display("_instr[31] %b", _instr[31]);
        // imm = { {12{_instr[31]}}, _instr[31], _instr[19:12], _instr[20], _instr[30:21], 1'b0 };
        imm = { {20{_instr[31]}}, _instr[31:20] };
        $display("imm %b %0d", imm, imm);
        write_reg(_rd_sel, pc + 4);
        next_pc = imm + pc;
        $display("pc %0d next_pc %0d", pc, next_pc);
        next_pc = imm + _rs1_data;
        $display("pc %0d next_pc %0d", pc, next_pc);
        $display("JALR storing %0d in x%0d", pc + 4, _rd_sel);
        read_next_instr(next_pc);
    endtask

    task op_store(input [addr_width - 1:0] _addr);
        $display("%0d STORE addr %0d <= %0d", pc, _addr, c1_rs2_data);
        `assert_known(_addr);
        case (_addr)
            1000: begin
                // write_out(regs[c1_rs2]);
                write_out(c1_rs2_data);
                $display("OUT %0d", c1_rs2_data);
                // immediately jump to next instruction, since not a real store...
                read_next_instr(pc + 4);
            end
            1004: begin
                $display("%0d 1004: HALT", pc);
                halt = 1;
            end
            1008: begin
                write_float(c1_rs2_data);
                $display("OUTR %0f", c1_rs2_data);
                read_next_instr(pc + 4);
            end
            default: begin
                // $display("default");
                // first write to memory; in C2 we will load next instruction
                write_mem(c1_rs1_data + c1_store_offset, c1_rs2_data);
                next_state = C2;
            end
        endcase
    endtask

    task instr_c1();
        // $display(
        //     "instr_c1 c1_op=%0d mem_rd_data=%b rs1_data=%0d rs2_data=%0d rd_sel=%0d c1_store_offset=%0d",
        //     c1_op, mem_rd_data, c1_rs1_data, c1_rs2_data, c1_rd_sel, c1_store_offset);
        // $strobe("strobe instr_c1 c1_op=%0d mem_rd_data=%b", c1_op, mem_rd_data);
        halt = 0;
        `assert_known(c1_op);
        case (c1_op)
            OPIMM: begin
                // $display("c1.OPIMM");
                op_imm(c1_funct3, c1_rd_sel, c1_rs1_sel, c1_i_imm);
            end
            LOAD: begin
                // $display("c1.LOAD c1_rs1=%0d regs[c1_rs1]=%0d c1_load_offset=%0d", c1_rs1_sel, regs[c1_rs1_sel], c1_load_offset);
                $display("%0d LOAD.C1  <= addr %0d", pc, c1_rs1_data + c1_load_offset);
                // read from memory
                // lw rd, offset(rs1)
                read_mem(c1_rs1_data + c1_load_offset);
                next_state = C2;
            end
            STORE: begin
                // $display("STORE");
                // write to memory
                // sw rs2, offset(rs1)
                op_store(c1_rs1_data + c1_store_offset);
            end
            BRANCH: begin
                // $display("c1.BRANCH");
                // e.g. beq rs1, rs2, offset
                op_branch(c1_funct3, c1_rs1_sel, c1_rs2_sel, c1_branch_offset);
            end
            OPFP: begin
                op_fp(c1_instr, c1_rd_sel, c1_rs1_data, c1_rs2_data);
            end
            OP: begin
                // $display("c1.OP");
                op_op(c1_op_funct, c1_rd_sel, c1_rs1_sel, c1_rs2_sel);
            end
            LUI: begin
                $display("c1.LUI");
                op_lui(c1_instr, c1_rd_sel);
            end
            AUIPC: begin
                $display("c1.AUIPC");
                op_auipc(c1_instr, c1_rd_sel);
            end
            JAL: begin
                $display("c1.JAL");
                op_jal(c1_instr, c1_rd_sel);
            end
            JALR: begin
                $display("c1.JALR");
                op_jalr(c1_instr, c1_rd_sel, c1_rs1_data);
            end
            default: begin
                $display("default: HALT c1_op %b", c1_op);
                halt = 1;
            end
        endcase
    endtask

    task instr_c2();
        // $display("C2 pc=%0d op=%0d %0b", pc, c2_op, c2_op);
        `assert_known(c2_op);
        case (c2_op)
            LOAD: begin
                // $display("C2.load mem_ack=%0b", mem_ack);
                `assert_known(mem_ack);
                if(mem_ack) begin
                    // $display("C2.load next c2_rd_sel=%0d mem_rd_data=%0d", c2_rd_sel, mem_rd_data);
                    $display("%0d LOAD.C2 x%0d <= %0d", pc, c2_rd_sel, mem_rd_data);
                    write_reg(c2_rd_sel, mem_rd_data);
                    read_next_instr(pc + 4);
                end
            end
            STORE: begin
                // $display("C2.store mem_ack %0b", mem_ack);
                `assert_known(mem_ack);
                if(mem_ack) begin
                    read_next_instr(pc + 4);
                end
            end
            OPFP: begin
                op_fp_c2(c2_instr, c2_rd_sel);
            end
            OP: begin
                // $display("OP.C2 op_funct=%0d", c2_op_funct);
                `assert_known(c2_op_funct);
                case(c2_op_funct)
                    MUL: begin
                        // $display("MUL.C2");
                        `assert_known(mul_ack);
                        if(mul_ack) begin
                            $display("MUL.C2 x%0d <=", c2_rd_sel);
                            wr_reg_data = mul_out;
                            wr_reg_sel = c2_rd_sel;
                            wr_reg_req = 1;
                            read_next_instr(pc + 4);
                        end 
                    end
                    DIVU: begin
                        // $display("DIVU.C2 div busy=%0b div_wr_reg_req=%0b", div_busy, div_wr_reg_req);
                        `assert_known(div_wr_reg_req);
                        if(div_wr_reg_req) begin
                            // go to next instruction
                            // well, lets read the result for now
                            $display("DIVU.C2 x%0d <= %0d", div_wr_reg_sel, div_wr_reg_data);
                            div_wr_reg_ack = 1;
                            write_reg(div_wr_reg_sel, div_wr_reg_data);
                            read_next_instr(pc + 4);
                        end
                    end
                    REMU: begin
                        // $display("REMU.C2 div busy=%0b div_wr_reg_req=%0b", div_busy, div_wr_reg_req);
                        `assert_known(div_wr_reg_req);
                        if(div_wr_reg_req) begin
                            // go to next instruction
                            // well, lets read the result for now
                            $display("REMU.C2 x%0d <= %0d", div_wr_reg_sel, div_wr_reg_data);
                            div_wr_reg_ack = 1;
                            write_reg(div_wr_reg_sel, div_wr_reg_data);
                            read_next_instr(pc + 4);
                        end
                    end
                    default: begin
                    end
                endcase
            end
            default: begin
                $display("WARNING got unrecognized op in c2 %0d", c2_op);
            end
        endcase
    endtask

    // always @(*) begin
    always @(mem_rd_data, div_wr_reg_req, c2_instr, state, pc, mem_ack, fadd_ack, mul_ack, fmul_ack) begin
        // $display("t=%0d proc.comb mem_rd_data=%0d div_wr_reg_req=%0d c2_instr=%0h state=%0d pc=%0d mem_ack=%0d",
        //     $time, mem_rd_data, div_wr_reg_req, c2_instr, state, pc, mem_ack
        // );
    // always_comb begin
        // $display("t=%0d proc.comb state=%0d pc=%0d",
        //     $time, state, pc
        // );

        halt = 0;
        out = '0;
        outen = 0;
        outflen = 0;

        mem_addr = '0;
        mem_rd_req = 0;
        mem_wr_req = 0;
        mem_wr_data = '0;

        next_pc = pc;
        next_state = state;

        div_wr_reg_ack = 0;

        c2_instr_next = c2_instr;

        wr_reg_sel = '0;
        wr_reg_data = '0;
        wr_reg_req = 0;

        n_div_req = 0;
        n_div_r_quot_sel = '0;
        n_div_r_mod_sel = '0;
        n_div_rs1_data = '0;
        n_div_rs2_data = '0;

        c1_instr = mem_rd_data;

        c1_op = c1_instr[6:0];
        c1_rd_sel = c1_instr[11:7];
        c1_rs1_sel = c1_instr[19:15];
        c1_rs2_sel = c1_instr[24:20];
        c1_funct3 = c1_instr[14:12];
        c1_imm1 = c1_instr[31:25];

        c1_rs1_data = regs[c1_rs1_sel];
        c1_rs2_data = regs[c1_rs2_sel];

        c1_store_offset = {{20{c1_instr[31]}}, c1_instr[31:25], c1_instr[11:7]};
        c1_load_offset = {{20{c1_instr[31]}}, c1_instr[31:20]};
        c1_i_imm = {{20{c1_instr[31]}}, c1_instr[31:20]};
        c1_branch_offset = {{20{c1_instr[31]}}, c1_instr[31], c1_instr[7], c1_instr[30:25], c1_instr[11:8]};
        c1_op_funct = {c1_instr[31:25], c1_instr[14:12]};

        x1 = regs[1];

        c2_op = c2_instr[6:0];
        c2_op_funct = {c2_instr[31:25], c2_instr[14:12]};
        c2_rd_sel = c2_instr[11:7];

        n_fadd_req = 0;
        n_fadd_a = '0;
        n_fadd_b = '0;

        n_fmul_req = 0;
        n_fmul_a = '0;
        n_fmul_b = '0;

        n_mul_req = 0;
        n_mul_a = '0;
        n_mul_b = '0;

        // if(~rst) begin
        //     $display(
        //         "t=%0d proc.comb state=%0d pc=%0d c1_op=%0d mem_rd_data=%0d mem_wr_req=%0b mem_rd_req=%0b mem_ack=%0b regs[1]=%0d",
        //         $time, state, pc, c1_op, mem_rd_data, mem_wr_req, mem_rd_req, mem_ack, regs[1]);
        // end
        `assert_known(state);
        case(state)
            C0: begin
                // if(~rst) begin
                //     $display("comb C0");
                // end
                read_next_instr(pc);
            end
            C1: begin
                // $display("comb C1 instr=%h", c1_instr);
                mem_rd_req = 0;
                if(mem_ack) begin
                    // $display("in mem_ack mem_ack=%0d mem_rd_data=%0d", mem_ack, mem_rd_data);
                    instr_c1();
                    c2_instr_next = mem_rd_data;
                end
            end
            C2: begin
                // $display("Comb.C2");
                instr_c2();
            end
            default: begin
                $display("Comb.default halt=1");
                halt = 1;
            end
        endcase
        // if(~rst) begin
        //     $display("comb end mem_rd_req=%0b",mem_rd_req);
        // end
    end

    always @(posedge clk or posedge rst) begin
        // assert(~$isunknown(rst));
        if (rst) begin
            // $display("proc.rst state=%0d n_state=%0d pc=%0d n_pc=%0d", state, next_state, pc, next_pc);
            pc <= 0;
            state <= C0;
            regs[0] <= '0;

            c2_instr <= '0;

            div_req <= '0;
            div_r_quot_sel <= '0;
            div_r_mod_sel <= '0;
            div_rs1_data <= '0;
            div_rs2_data <= '0;

            fadd_req <= 0;
            fadd_a <= '0;
            fadd_b <= '0;

            fmul_req <= 0;
            fmul_a <= '0;
            fmul_b <= '0;

            mul_req <= '0;
            mul_a <= '0;
            mul_b <= '0;
        end else begin
            // $display("proc.run pc=%0d state=%0d mem_ack=%0d mem_rd_data=%0d %h", pc, state, mem_ack, mem_rd_data, mem_rd_data);
            // $display(
            //     "t=%0d proc.ff mem_addr %0d mem_wr_data %0d mem_rd_data %0d mem_wr_req %b mem_rd_req  %b mem_ack %b mem_busy %b",
            //     $time,
            //     mem_addr,     mem_wr_data,    mem_rd_data,    mem_wr_req,   mem_rd_req,    mem_ack,   mem_busy);
            // $display("ff tick t=%0d clk=%0b next_pc=%0d next_state=%0d", $time, clk, next_pc, next_state);
            pc <= next_pc;
            state <= next_state;
            c2_instr <= c2_instr_next;

            `assert_known(wr_reg_req);
            if (wr_reg_req && wr_reg_sel != 0) begin
                regs[wr_reg_sel] <= wr_reg_data;
            end

            div_req <=        n_div_req;
            div_r_quot_sel <= n_div_r_quot_sel;
            div_r_mod_sel <=  n_div_r_mod_sel;
            div_rs1_data <=   n_div_rs1_data;
            div_rs2_data <=   n_div_rs2_data;

            fadd_req <= n_fadd_req;
            fadd_a <= n_fadd_a;
            fadd_b <= n_fadd_b;

            fmul_req <= n_fmul_req;
            fmul_a <= n_fmul_a;
            fmul_b <= n_fmul_b;

            mul_req <= n_mul_req;
            mul_a <= n_mul_a;
            mul_b <= n_mul_b;
        end
    end
endmodule
