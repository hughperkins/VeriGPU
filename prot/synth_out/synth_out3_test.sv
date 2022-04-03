module synth_out3_test();
    reg clk;
    reg rst;
    reg req;
    reg [3:0] cnt;

    synth_out3 synth_out_(
        .clk(clk),
        .cnt(cnt),
        .rst(rst),
        .req(req)
    );

    task pos();
        $display("  +");
        #5 clk = 1;
    endtask

    task neg();
        $display("-");
        #5 clk = 0;
    endtask

    task tick();
        $display("-");
        #5 clk = 0;
        $display("  +");
        #5 clk = 1;
    endtask

    initial begin
        rst = 0;
        req = 0;
        tick();
        tick();

        rst = 1;
        $monitor("req=%0b cnt=%0d", req, cnt);

        tick();
        // assert(~out);

        tick();
        // assert(~out);
        req = 1;

        neg();
        pos();
        req = 0;

        neg();
        // assert(out);
        pos();

        // assert(out);

        tick();

        tick();
        tick();

        $finish;
    end
endmodule
