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
    reg [7:0] instruction;
    typedef enum {
        RESET,
        AWAITING_INSTR,
        GOT_INSTR
    } e_state;
    // reg [4:0] state;
    //reg instruction_done;
    //reg instruction_rdy;
    //reg instruction_fetched;

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
                        instruction <= mem_read_data;
                        op <= mem_read_data[11:8];
                        reg_select <= mem_read_data[15:12];
                        p1 <= mem_read_data[7:0];
                        state <= GOT_INSTR;
                    end
                end
                GOT_INSTR: begin
                    case (op)
                       4'h1: begin
                           // out immediate
                           out[7:0] <= p1;
                           outen <= 1;
                            pc <= pc + 1;
                            mem_read_addr <= pc + 1;
                            state <= AWAITING_INSTR;
                       end
                       4'h2: begin
                           // outloc
                           if (mem_read_addr == p1 >> 1) begin
                               out <= mem_read_data;
                               outen <= 1;
                                pc <= pc + 1;
                                mem_read_addr <= pc + 1;
                                state <= AWAITING_INSTR;
                           end else begin
                               mem_read_addr <= p1 >> 1;
                               // pc <= pc - 1;
                            end
                       end
                       4'h3: begin
                          // li
                          regs[reg_select] <= p1;
                            pc <= pc + 1;
                            mem_read_addr <= pc + 1;
                            state <= AWAITING_INSTR;
                       end
                       4'h4: begin
                           // outr
                           out[7:0] <= regs[reg_select];
                           outen <= 1;
                            pc <= pc + 1;
                            mem_read_addr <= pc + 1;
                            state <= AWAITING_INSTR;
                       end
                       default: out <= '0;
                    endcase
                end
            endcase
            //if (~instruction_fetched) begin
            //    read_mem_addr <= pc;
            //    instruction_fetched <= 1;
            //end else if()
            //pc <= pc + 1;
        end
    end
    // assign mem_read_addr = pc;
    // assign op = mem_read_data[11:8];
    // assign reg_select = mem_read_data[15:12];
    // assign p1 = mem_read_data[7:0];
    assign x1 = regs[1];
endmodule
