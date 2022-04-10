// `timescale 1ns/10ps

module core_driver();
    parameter mem_load_size = 256;
    parameter out_size = 128;

    reg rst;
    reg clk;

    wire [31:0] out;
    wire outen;
    wire outflen;

    wire core1_mem_rd_req;
    wire core1_mem_wr_req;

    wire [addr_width - 1:0] core1_mem_addr;
    wire [data_width - 1:0] core1_mem_rd_data;
    wire [data_width - 1:0] core1_mem_wr_data;

    wire core1_mem_busy;
    wire core1_mem_ack;

    reg contr_mem_wr_en;
    reg contr_mem_rd_en;
    reg [addr_width - 1:0] contr_mem_wr_addr;
    reg [data_width - 1:0] contr_mem_wr_data;
    reg [addr_width - 1:0] contr_mem_rd_addr;
    reg [data_width - 1:0] contr_mem_rd_data;
    reg contr_mem_rd_ack;

    reg contr_core1_ena;
    reg contr_core1_clr;
    reg contr_core1_set_pc_req;
    reg [data_width - 1:0] contr_core1_set_pc_addr;
    wire contr_core1_halt;

    // reg core1_halt;

    // wire [31:0] pc;
    // wire [6:0] op;
    // wire [4:0] rd;
    // wire [6:0] imm1;
    // wire [31:0] x1;
    // wire [4:0] state;

    // reg [31:0] oob_wr_addr;
    // reg [31:0] oob_wr_data;
    // reg oob_wen;

    reg [31:0] mem_load [mem_load_size];

    reg [31:0] outmem [out_size];
    reg [out_size]outtype ;
    reg [$clog2(out_size) - 1:0] outpos;
    reg halt;

    reg [63:0] double;

    reg [31:0] t_at_reset;
    reg [31:0] cycle_count;

    global_mem_controller global_mem_controller_(
        .clk(clk),
        .rst(rst),
        // .ena(ena),

        .core1_addr(core1_mem_addr),
        .core1_wr_req(core1_mem_wr_req),
        .core1_rd_req(core1_mem_rd_req),
        .core1_rd_data(core1_mem_rd_data),
        .core1_wr_data(core1_mem_wr_data),
        .core1_busy(core1_mem_busy),
        .core1_ack(core1_mem_ack),

        .contr_wr_en(contr_mem_wr_en),
        .contr_rd_en(contr_mem_rd_en),
        .contr_wr_addr(contr_mem_wr_addr),
        .contr_wr_data(contr_mem_wr_data),
        .contr_rd_addr(contr_mem_rd_addr),
        .contr_rd_data(contr_mem_rd_data),
        .contr_rd_ack(contr_mem_rd_ack)
    );

    core core1(
        .rst(rst),
        .clk(clk),
        .clr(contr_core1_clr),
        .ena(contr_core1_ena),
        .set_pc_req(contr_core1_set_pc_req),
        .set_pc_addr(contr_core1_set_pc_addr),

        .outflen(outflen),
        .out(out),
        .outen(outen),

        .halt(contr_core1_halt),

        .mem_addr(core1_mem_addr),
        .mem_rd_data(core1_mem_rd_data),
        .mem_wr_data(core1_mem_wr_data),
        .mem_ack(core1_mem_ack),
        .mem_busy(core1_mem_busy),
        .mem_rd_req(core1_mem_rd_req),
        .mem_wr_req(core1_mem_wr_req)
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
        rst <= 0;
        contr_core1_ena <= 0;
        contr_core1_clr <= 0;

        // ena <= 0;
        // oob_wen <= 0;
        #10
        #5;

        rst <= 1;
        contr_core1_set_pc_req <= 1;
        contr_core1_set_pc_addr <= 128;

        #10
        contr_core1_set_pc_req <= 0;

        // write the program into global memory
        for(int i = 0; i < 255; i++) begin
            contr_mem_wr_en <= 1;
            contr_mem_wr_addr <= (i << 2) + 128;
            // $display("mem_load[%0d]= %0d", i, mem_load[i]);
            contr_mem_wr_data <= mem_load[i];
            // oob_wen <= 1;
            // oob_wr_addr <= (i << 2) + 128;
            // oob_wr_data <= mem_load[i];
            #10;
        end
        contr_mem_wr_en <= 0;
        // oob_wen <= 0;
        outpos <= 0;
        #10;
        $display("");
        $display("===========================================");
        $display("========== turning on enable ==============");
        $display("");
        // rst = 0;
        contr_core1_ena <= 1;
        t_at_reset = $time;

        $monitor(
            "t=%0d core_driver =============",
            $time());

        // #500
        // $finish;

        // rst = 1;
        // #1 rst = 0;

        // while(~halt && $time < 4040) begin
        // while(~halt && $time - t_at_reset < 3940) begin
        while(~contr_core1_halt && $time - t_at_reset < 400000) begin
        // while(~halt && $time - t_at_reset < 200000) begin
        // while(~halt && $time - t_at_reset < 6000) begin
        // while(~halt && $time - t_at_reset < 10000) begin
        // while(~halt && $time - t_at_reset < 50000) begin
        // while(~halt && $time - t_at_reset < 1200) begin
        // while(~halt && $time - t_at_reset < 50) begin
            #10;
        end

        $display("t=%0d core_driver.halt %0b", $time, contr_core1_halt);
        cycle_count = ($time - t_at_reset) / 10;

        $display("t=%0d core_driver monitor outpos %0d", $time, outpos);
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
