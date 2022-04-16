/*

python verigpu/timing.py --in-verilog src/const.sv src/int_div_regfile.sv src/apu.sv --top-module apu

Propagation delay is between any pair of combinatorially connected
inputs and outputs, drawn from:
    - module inputs
    - module outputs,
    - flip-flop outputs (treated as inputs), and
    - flip-flop inputs (treated as outputs)

max propagation delay: 45.8 nand units

*/

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

module apu(
    input clk, rst,

    // incoming request
    input [data_width - 1:0] rs1,
    input [data_width - 1:0] rs2,
    input [reg_sel_width - 1:0] rd_sel,
    input req,
    input [9:0] funct,
    output reg busy,

    // connection to reg file
    output reg apu_wr_req,
    output reg [reg_sel_width - 1:0] apu_wr_sel,
    output reg [data_width - 1:0] apu_wr_data
);
    // division
    // reg div_rs1;
    // reg div_rs2;
    // reg div_rd_sel;
    // reg div_
    reg [4:0] div_wr_sel;
    reg [31:0] div_wr_data;
    reg div_req;
    reg div_wr_req;
    reg div_busy;

    int_div_regfile int_div_regfile_(
        .clk(clk),
        .rst(rst),
        .req(div_req),
        .busy(div_busy),
        .r_quot_sel(rd_sel),
        .r_mod_sel(5'b0),
        .a(rs1),
        .b(rs2),
        .rf_wr_sel(div_wr_sel),
        .rf_wr_data(div_wr_data),
        .rf_wr_req(div_wr_req)
    );

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            apu_wr_req <= 0;
            busy <= 0;
            div_req <= 0;
        end else begin
            busy <= 0;
            apu_wr_req <= 0;
            div_req <= 0;

            // process incoming requests
            // $display("funct %b div_req %0b div_busy %0b div_wr_req %0b div_wr_sel %0d div_wr_data %0d", funct, div_req, div_busy, div_wr_req, div_wr_sel, div_wr_data);
            if(req) begin
                case(funct)
                    DIVU: begin
                        if (div_busy) begin
                            busy <= 1;
                        end else begin
                            // div <= rd_sel;
                            div_req <= 1;
                        end
                    end
                endcase
            end

            // write data back to register for finished tasks
            if(div_wr_req) begin
                apu_wr_data <= div_wr_data;
                apu_wr_sel <= div_wr_sel;
                apu_wr_req <= 1;
            end
        end
    end
endmodule
