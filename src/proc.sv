// represents processor
module proc(
    input rst, clk,
    output reg [15:0] out,
    output reg [3:0] op,
    output reg [3:0] reg_select,
    output reg [7:0] p1,
    output [7:0] x1,
    output reg [15:0] pc,
    output reg [4:0] state,
    output reg outen,

    output reg mem_we,
    output reg [15:0] mem_read_addr,
    output reg [15:0] mem_write_addr,
    input [15:0] mem_read_data,
    output reg [15:0] mem_write_data
);
    reg [7:0] regs[16];
    reg [15:0] instruction;
    typedef enum bit[4:0] {
        RESET,
        AWAITING_INSTR,
        GOT_INSTR
    } e_state;
    // reg [4:0] state;
    //reg instruction_done;
    //reg instruction_rdy;
    //reg instruction_fetched;

    wire [3:0] c1_op;
    wire [3:0] c1_reg_select;
    wire [7:0] c1_p1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= '0;
            pc <= '0;
            state <= RESET;
            //instruction_done <= 0;
            //instruction_rdy <= 0;
            //instruction_fetched <= 0;
        end
        else begin
            out <= '0;
            mem_we <= 0;
            outen <= 0;
            case(state)
                RESET: begin
                    mem_read_addr <= pc;
                    state <= AWAITING_INSTR;
                end
                AWAITING_INSTR: begin
                    if(mem_read_addr == pc) begin
                        case (c1_op)
                           4'h1: begin
                               // out immediate
                               out[7:0] <= c1_p1;
                               outen <= 1;
                                pc <= pc + 1;
                                mem_read_addr <= pc + 1;
                                state <= AWAITING_INSTR;
                           end
                           4'h2: begin
                               // outloc
                              // if (mem_read_addr == {8'b0, 1'b0, p1[7:1]}) begin
                                //   out <= mem_read_data;
                                  // outen <= 1;
                                    //pc <= pc + 1;
                                   // mem_read_addr <= pc + 1;
                                   // state <= AWAITING_INSTR;
                               //end else begin
                                   mem_read_addr <= {8'b0, 1'b0, c1_p1[7:1]};
                                   state <= GOT_INSTR;
                                   // pc <= pc - 1;
                                //end
                           end
                           4'h3: begin
                              // li
                              regs[c1_reg_select] <= c1_p1;
                                pc <= pc + 1;
                                mem_read_addr <= pc + 1;
                                state <= AWAITING_INSTR;
                           end
                           4'h4: begin
                               // outr
                               out[7:0] <= regs[c1_reg_select];
                               outen <= 1;
                                pc <= pc + 1;
                                mem_read_addr <= pc + 1;
                                state <= AWAITING_INSTR;
                           end
                           default: out <= '0;
                        endcase

                        instruction <= mem_read_data;
                        op <= mem_read_data[11:8];
                        reg_select <= mem_read_data[15:12];
                        p1 <= mem_read_data[7:0];
                        // state <= GOT_INSTR;
                    end
                end
                GOT_INSTR: begin
                    case (op)
                        4'h2: begin
                            // outloc
                           if (mem_read_addr == {8'b0, 1'b0, p1[7:1]}) begin
                                out <= mem_read_data;
                                outen <= 1;
                                pc <= pc + 1;
                                mem_read_addr <= pc + 1;
                                state <= AWAITING_INSTR;
                           end
                        end
                    endcase
                end
                default: out <= '0;
            endcase
        end
    end
    assign c1_op = mem_read_data[11:8];
    assign c1_reg_select = mem_read_data[15:12];
    assign c1_p1 = mem_read_data[7:0];
    assign x1 = regs[1];
endmodule
