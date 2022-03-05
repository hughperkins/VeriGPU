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

    // output reg mem_we,
    output reg [15:0] mem_addr,
    input [15:0] mem_rd_data,
    output reg [15:0] mem_wr_data,
    output reg mem_wr_req,
    output reg mem_rd_req,
    input mem_ack,
    input mem_busy
);
    reg [7:0] regs[16];
    reg [15:0] instruction;
    typedef enum bit[4:0] {
        RESET,
        AWAITING_INSTR,
        GOT_INSTR,
        GOT_INSTR2
    } e_state;

    wire [3:0] c1_op;
    wire [3:0] c1_reg_select;
    wire [7:0] c1_p1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= '0;
            pc <= '0;
            state <= RESET;
        end
        else begin
            out <= '0;
            // mem_we <= 0;
            mem_rd_req <= 0;
            mem_wr_req <= 0;
            outen <= 0;
            case(state)
                RESET: begin
                    mem_addr <= pc;
                    mem_rd_req <= 1'b1;
                    state <= AWAITING_INSTR;
                end
                AWAITING_INSTR: begin
                   // $display("AWAITINGINSTR ack=%d addr %d req %d busy", mem_ack, mem_addr, mem_rd_req, mem_busy);
                    mem_rd_req <= 1'b0;
                    if(mem_ack) begin
                        case (c1_op)
                           4'h1: begin
                               // out immediate
                               out[7:0] <= c1_p1;
                               outen <= 1;
                                pc <= pc + 1;
                                mem_addr <= pc + 1;
                                mem_rd_req <= 1'b1;
                                state <= AWAITING_INSTR;
                           end
                           4'h2: begin
                               // outloc
                                   mem_addr <= {8'b0, 1'b0, c1_p1[7:1]};
                                    mem_rd_req <= 1'b1;
                                   state <= GOT_INSTR;
                                //end
                           end
                           4'h3: begin
                              // li
                              regs[c1_reg_select] <= c1_p1;
                                pc <= pc + 1;
                                mem_addr <= pc + 1;
                                mem_rd_req <= 1'b1;
                                state <= AWAITING_INSTR;
                           end
                           4'h4: begin
                               // outr
                               out[7:0] <= regs[c1_reg_select];
                               outen <= 1;
                                pc <= pc + 1;
                                mem_addr <= pc + 1;
                                mem_rd_req <= 1'b1;
                                state <= AWAITING_INSTR;
                           end
                           default: out <= '0;
                        endcase

                        instruction <= mem_rd_data;
                        op <= mem_rd_data[11:8];
                        reg_select <= mem_rd_data[15:12];
                        p1 <= mem_rd_data[7:0];
                    end
                end
                //GOT_INSTR: begin
                //    if(mem_addr == pc) begin
                //    end
                //end
                GOT_INSTR: begin
                    case (op)
                        4'h2: begin
                            // outloc
                           if (mem_addr == {8'b0, 1'b0, p1[7:1]}) begin
                                out <= mem_rd_data;
                                outen <= 1;
                                pc <= pc + 1;
                                mem_addr <= pc + 1;
                                state <= AWAITING_INSTR;
                           end
                        end
                    endcase
                end
                default: out <= '0;
            endcase
        end
    end
    assign c1_op = mem_rd_data[11:8];
    assign c1_reg_select = mem_rd_data[15:12];
    assign c1_p1 = mem_rd_data[7:0];
    assign x1 = regs[1];
endmodule
