// `timescale 1ns/10ps

module comp_driver();
    parameter mem_load_size = 256;
    parameter out_size = 128;

    reg rst;
    reg clk;
    reg ena;

    wire [31:0] out;
    wire outen;
    wire outflen;

    wire [31:0] pc;
    wire [6:0] op;
    wire [4:0] rd;
    wire [6:0] imm1;
    wire [31:0] x1;
    wire [4:0] state;

    reg [31:0] oob_wr_addr;
    reg [31:0] oob_wr_data;
    reg oob_wen;

    reg [31:0] mem_load [mem_load_size];

    reg [31:0] outmem [out_size];
    reg [out_size]outtype ;
    reg [$clog2(out_size) - 1:0] outpos;
    reg halt;

    reg [63:0] double;

    reg [31:0] t_at_reset;
    reg [31:0] cycle_count;

    comp comp1(
        .clk(clk), .rst(rst), .ena(ena),
        .pc(pc), .op(op), .rd(rd),
        .x1(x1), .imm1(imm1), .state(state),
        .out(out), .outen(outen), .outflen(outflen),
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
        if ((outen | outflen) & outpos < out_size - 1 ) begin
            outmem[outpos] <= out;
            outtype[outpos] <= outflen;
            outpos <= outpos + 1;
        end
    end

    function [63:0] bitstosingle(input [31:0] s);
        bitstosingle = { s[31], s[30], {3{~s[30]}}, s[29:23], s[22:0], {29{1'b0}} };
    endfunction

    initial begin
        $readmemh("build/prog.hex", mem_load);
        rst <= 1;
        ena <= 0;
        oob_wen <= 0;
        #10
        rst <= 0;
        #10
        #5;

        for(int i = 0; i < 255; i++) begin
            oob_wen <= 1;
            oob_wr_addr <= i;
            oob_wr_data <= mem_load[i];
            #10;
        end
        oob_wen <= 0;
        outpos <= 0;
        #10;
        $display("");
        $display("===========================================");
        $display("========== turning on enable ==============");
        $display("");
        // rst = 0;
        ena <= 1;
        t_at_reset = $time;

        $monitor(
            "t=%0d comp_driver rst=%b pc=%0d =============",
            $time(), rst, pc);

        // #500
        // $finish;

        // rst = 1;
        // #1 rst = 0;

        // while(~halt && $time < 4040) begin
        // while(~halt && $time - t_at_reset < 3940) begin
        while(~halt && $time - t_at_reset < 400000) begin
        // while(~halt && $time - t_at_reset < 200000) begin
        // while(~halt && $time - t_at_reset < 6000) begin
        // while(~halt && $time - t_at_reset < 10000) begin
        // while(~halt && $time - t_at_reset < 50000) begin
        // while(~halt && $time - t_at_reset < 1200) begin
        // while(~halt && $time - t_at_reset < 50) begin
            #10;
        end

        $display("t=%0d comp_driver.halt %0b", $time, halt);
        cycle_count = ($time - t_at_reset) / 10;

        $display("t=%0d comp_driver monitor outpos %0d", $time, outpos);
        $display("");
        for(int i = 0; i < outpos; i++) begin
            if (outtype[i]) begin
                double = bitstosingle(outmem[i]);
                $display("out.s %0d %b %f", i, outmem[i], $bitstoreal(double));
            end else begin
                $display("out %0d %b %h %0d", i, outmem[i], outmem[i], outmem[i]);
            end
        end
        $monitor("");
        $display("Cycle count is number of clock cycles from reset going low to halt received.");
        $display("cycle_count %0d", cycle_count);
        $display("");
        $finish();
    end
endmodule
