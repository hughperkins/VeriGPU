// driver for test_test_timings

module test_test_timings_test();
    reg clk, rst;
    reg req, out;
    reg clr;

    reg autoclock;

    test_test_timings test_test_timings1(
        .rst(rst),
        .clk(clk),
        .req(req),
        .out(out),
        .clr(clr)
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
        autoclock = 0;
        forever begin
            // #5
            #5;
            if(autoclock) begin
                clk = ~clk;
                if(clk) begin
                    $display("  + (auto)");
                end else begin
                    $display("-   (auto)");
                end
            end
        end
    end

    initial begin
        $monitor("t=%0d rst=%0d req=%0d out=%0d", $time, rst, req, out);
        // 0
        rst <= 1;
        req <= 0;
        clr <= 0;
        tick();
        // 10
        rst <= 0;
        tick();
        // 20
        `assert(~out);
        tick();
        // 30
        `assert(~out);
        req <= 1;
        tick();
        // 40
        neg();
        // 45
        `assert(out);
        pos();
        // 50
        req <= 0;
        clr <= 1;
        tick();
        clr <= 0;
        neg();
        `assert(~out);
        pos();
        autoclock = 1;
        #5
        assert(clk == 0);
        #5
        assert(clk == 1);

        req <= 1;
        #10;
        #5;
        `assert(out);

        #100 $finish;
    end
endmodule
