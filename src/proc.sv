// represents processor
module proc(
    input rst, clk,
    output reg [data_width - 1:0] out,
    output reg outen,
    output reg outflen,

    output reg [data_width - 1:0] x1,
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

    reg [6:0] c1_op;
    reg [2:0] c1_funct3;
    reg [9:0] c1_op_funct;
    reg [reg_sel_width - 1:0] c1_rd_sel;
    reg [reg_sel_width - 1:0] c1_rs1_sel;
    reg [reg_sel_width - 1:0] c1_rs2_sel;
    reg [data_width - 1:0] c1_rs1_data;
    reg [data_width - 1:0] c1_rs2_data;
    reg [6:0] c1_imm1;

    reg signed [addr_width - 1:0] c1_store_offset;
    reg signed [addr_width - 1:0] c1_load_offset;
    reg signed [data_width - 1:0] c1_i_imm;
    reg signed [addr_width - 1:0] c1_branch_offset;
    reg [instr_width - 1:0] c1_instr;

    reg [6:0] c2_op;
    reg [9:0] c2_op_funct;
    reg [reg_sel_width - 1:0] c2_rd_sel;
    reg [reg_sel_width - 1:0] c2_rs1_sel;
    reg [reg_sel_width - 1:0] c2_rs2_sel;
    reg [6:0] c2_imm1;

    reg [reg_sel_width - 1:0] wr_reg_sel;
    reg [data_width - 1:0] wr_reg_data;
    reg wr_reg_req;

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
            BEQ: if (c1_rs1_data == c1_rs2_data) branch = 1;
            BNE: if (c1_rs1_data != c1_rs2_data) branch = 1;
            default: begin end
        endcase

        if (branch) begin
            read_next_instr(pc + {_offset[30:0], 1'b0});
        end else begin
            read_next_instr(pc + 4);
        end
    endtask

    task op_op(input [9:0] _funct, input [4:0] _rd_sel, input [4:0] _rs1_sel, input [4:0] _rs2_sel);
        reg skip_advance_pc;

        wr_reg_req = 1;
        wr_reg_sel = _rd_sel;
        $display("op_op.c1 op_funct=%0d", _funct);
        skip_advance_pc = 0;
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
            DIVU: begin
                $display("DIVU.c1");
                if(~div_busy) begin
                    $display("sending req to div unit a=%0d b=%0d quot_sel=%0d", c1_rs1_data, c1_rs2_data, _rd_sel);
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
                    // wait for not busy I suppose...
                    $display("waiting for div unit to be free");
                end
            end
            REMU: begin
                $display("REMU.c1");
                if(~div_busy) begin
                    $display("sending req to div unit a=%0d b=%0d mod_sel=%0d", c1_rs1_data, c1_rs2_data, _rd_sel);
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
                    $display("waiting for div unit to be free");
                end
            end
            default: begin end
        endcase
        // $display("op regs[_rd]=%0d _rd=%0d regs[_rs1]=%0d regs[_rs2]=%0d", regs[_rd], _rd, regs[_rs1], regs[_rs2]);
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

    task op_store(input [addr_width - 1:0] _addr);
        $display("op_store addr %0d", _addr);
        case (_addr)
            1000: begin
                // write_out(regs[c1_rs2]);
                write_out(c1_rs2_data);
                $display(" store 1000 %0d", c1_rs2_data);
                // immediately jump to next instruction, since not a real store...
                read_next_instr(pc + 4);
            end
            1004: begin
                $display("1004: HALT");
                halt = 1;
            end
            1008: begin
                write_float(c1_rs2_data);
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
        $display("instr_c1 c1_op=%0d mem_rd_data=%b rs1_data=%0d c1_store_offset=%0d", c1_op, mem_rd_data, c1_rs1_data, c1_store_offset);
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
                read_mem(c1_rs1_data + c1_load_offset);
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
            OP: begin
                $display("OP.C2 op_funct=%0d", c2_op_funct);
                case(c2_op_funct)
                    DIVU: begin
                        $display("DIVU.C2 div busy=%0b div_wr_reg_req=%0b", div_busy, div_wr_reg_req);
                        if(div_wr_reg_req) begin
                            // go to next instruction
                            // well, lets read the result for now
                            $display("got write req from div, write ack");
                            div_wr_reg_ack = 1;
                            write_reg(div_wr_reg_sel, div_wr_reg_data);
                            read_next_instr(pc + 4);
                        end
                    end
                    REMU: begin
                        $display("REMU.C2 div busy=%0b div_wr_reg_req=%0b", div_busy, div_wr_reg_req);
                        if(div_wr_reg_req) begin
                            // go to next instruction
                            // well, lets read the result for now
                            $display("got write req from div, write ack");
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
            end
        endcase
    endtask

    // always @(mem_rd_data, div_wr_reg_req, c2_instr, state, pc, mem_ack) begin
    always @(*) begin
        $display("mem_rd_data=%0d div_wr_reg_req=%0d c2_instr=%0h state=%0d pc=%0d mem_ack=%0d",
            mem_rd_data, div_wr_reg_req, c2_instr, state, pc, mem_ack
        );
    // always_comb begin
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
                $display("Comb.default halt=1");
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
            // $display(
            //     "ff mem_addr %0d mem_wr_data %0d mem_rd_data %0d mem_wr_req %b mem_rd_req  %b mem_ack %b mem_busy %b",
            //     mem_addr,     mem_wr_data,    mem_rd_data,    mem_wr_req,   mem_rd_req,    mem_ack,   mem_busy);
            // $display("ff tick t=%0d clk=%0b next_pc=%0d next_state=%0d", $time, clk, next_pc, next_state);
            pc <= next_pc;
            state <= next_state;
            c2_instr <= c2_instr_next;
            if (wr_reg_req) begin
                regs[wr_reg_sel] <= wr_reg_data;
            end

            div_req <= n_div_req;
            div_r_quot_sel <= n_div_r_quot_sel;
            div_r_mod_sel <= n_div_r_mod_sel;
            div_rs1_data <= n_div_rs1_data;
            div_rs2_data <= n_div_rs2_data;
        end
    end
endmodule
