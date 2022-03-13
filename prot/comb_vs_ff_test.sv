module comb_vs_ff_test();
    reg clk;
    reg rst;
    reg [3:0] op;
    wire [data_width - 1:0] cnt;

    comb_vs_ff comb_vs_ff_(
        .clk(clk),
        .rst(rst),
        .op(op),
        .cnt(cnt)
    );

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%0d rst=%0b cnt=%0d", $time, rst, cnt);
        rst = 1;
        #10
        rst = 0;
        op = ADD;

        #40
        op = SUB;

        #30
        op = MUL2;

        #30
        op = ADDFIVE;

        #30
        op = DIV2;

        #30
        $finish;
    end
endmodule
