module comp_driver(
);
    reg rst;
    reg clk;

    wire [31:0] out;
    wire [31:0] pc;
    wire [6:0] op;
    wire [4:0] rd;
    wire [6:0] imm1;
    wire [31:0] x1;
    wire [4:0] state;
    wire outen;

    reg [31:0] oob_wr_addr;
    reg [31:0] oob_wr_data;
    reg oob_wen;

    reg [31:0] mem_load [256];

    reg [31:0] outmem [32];
    reg [4:0] outpos;
    reg halt;

    comp comp1(
        .clk(clk), .rst(rst),
        .pc(pc), .op(op), .rd(rd),
        .x1(x1), .imm1(imm1), .state(state),
        .out(out), .outen(outen),
        .oob_wr_addr(oob_wr_addr),
        .oob_wr_data(oob_wr_data),
        .oob_wen(oob_wen),
        .halt(halt)
    );

    initial begin
        clk = 1;
        forever #5 clk = ~clk;
    end
    always @(posedge clk) begin
        if (outen) begin
            outmem[outpos] <= out;
            outpos <= outpos + 1;
        end
    end
    initial begin
        $readmemh("build/{PROG}.hex", mem_load);
        for(int i = 0; i < 255; i++) begin
            #10
            oob_wen = 1;
            oob_wr_addr = i;
            oob_wr_data = mem_load[i];
        end
        #10
        oob_wen = 0;
        outpos = 0;
        #10

        $monitor(
            "t=%d rst=%b pc=%h, out=%h op=%h imm1=%h rd=%h x1=%h state=%d",
            $time(), rst, pc, out,  op,   imm1,   rd, x1, state);
        rst = 1;
        #10 rst = 0;

        while(~halt && clk < 100) begin
            #10;
        end

        // #1000
        for(int i = 0; i < outpos; i++) begin
            $display("out %h %h", i, outmem[i]);
        end
        $finish();
    end
endmodule
