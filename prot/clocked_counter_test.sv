module test_clocked_counter();
    reg rst, clk;
    reg [31:0] cnt;
    clocked_counter dut(.clk(clk), .rst(rst), .cnt(cnt));

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%0d rst=%0d cnt=%0d", $time, rst, cnt);
        rst = 1;
        #20 rst = 0;
        #100 $finish();
    end
endmodule
