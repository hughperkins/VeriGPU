module comp_driver();
    reg rst, clk;
    reg [15:0] out;
    reg [15:0] mem [256];
    reg [3:0] op;
    reg [3:0] reg_select;
    reg [7:0] p1;
    //reg [7:0] regs[16];
    reg [7:0] x1;

    reg [15:0] mem_read_addr, mem_write_addr;
    reg [15:0] mem_read_data, mem_write_data;
    reg [15:0] oob_write_addr;
    reg [15:0] oob_write_data;
    wire mem_we;
    reg oob_mem_wen;

    reg [15:0] pc;

    mem mem1(
        .clk(clk), .we(mem_we),

        .read_addr(mem_read_addr), .write_addr(mem_write_addr),
        .read_data(mem_read_data), .write_data(mem_write_data),

        .oob_write_addr(oob_write_addr), .oob_write_data(oob_write_data),
        .oob_wen(oob_mem_wen)
    );

    comp comp1(
        .rst(rst), .clk(clk), .out(out), .op(op), .p1(p1), .pc(pc),
        .reg_select(reg_select),
        .x1(x1),
        .mem_read_addr(mem_read_addr), .mem_write_addr(mem_write_addr),
        .mem_read_data(mem_read_data), .mem_write_data(mem_write_data),
        .mem_we(mem_we)
    );
    initial begin
        clk = 1;
        forever #5 clk = ~clk;
    end
    initial begin
        $readmemh("build/{PROG}.hex", mem);
        for(int i = 0; i < 255; i++) begin
            #10
            oob_mem_wen = 1;
            oob_write_addr = i;
            oob_write_data = mem[i];
        end
        #10
        oob_mem_wen = 0;
        #10

        $monitor(
            "t=%d rst=%b pc=%h, out=%h op=%h p1=%h rs=%h x1=%h",
            $time(), rst, pc, out,  op,   p1,   reg_select, x1);
        rst = 1;
        #10 rst = 0;
        #200 $finish();
    end
endmodule
