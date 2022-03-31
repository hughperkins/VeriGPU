module int_div_regfile_test();
    reg clk;
    reg rst;

    reg req;
    reg busy;

    reg [reg_sel_width - 1: 0] r_quot_sel;  // 0 means, dont write (i.e. x0)
    reg [reg_sel_width - 1: 0] r_mod_sel;   // 0 means, dont write  (i.e. x0)
    reg [data_width - 1:0] a;
    reg [data_width - 1:0] b;

    reg [reg_sel_width - 1:0] rf_wr_sel;
    reg [data_width - 1:0] rf_wr_data;
    reg rf_wr_req;
    reg rf_wr_ack;

    reg [31:0] cnt;

    int_div_regfile dut(
        .clk(clk),
        .rst(rst),
        .req(req),
        .busy(busy),
        .r_quot_sel(r_quot_sel),
        .r_mod_sel(r_mod_sel),
        .a(a),
        .b(b),
        .rf_wr_sel(rf_wr_sel),
        .rf_wr_data(rf_wr_data),
        .rf_wr_req(rf_wr_req),
        .rf_wr_ack(rf_wr_ack)
    );

    initial begin
        clk = 1;
        forever begin
            #5 clk=~clk;
        end
    end

    task test_div_zero(
        [data_width - 1:0] _a,
        [data_width - 1:0] _b,
        [data_width - 1:0] expected_quot,
        [data_width - 1:0] expected_mod
    );
        $display("");
        $display("==================");
        $display("test %0d / %0d => %0d r %0d", _a, _b, expected_quot, expected_mod);

        `assert(~busy);
        `assert(~rf_wr_ack);

        a <= _a;
        b <= _b;  // 81 r 37
        req <= 1;
        cnt <= 0;
        r_quot_sel <= 3;
        r_mod_sel <= 7;

        #10
        req <= 0;
        a <= 0;
        b <= 0;
        r_quot_sel <= 0;
        r_mod_sel <= 0;

        `assert(busy);
        `assert(rf_wr_req);
        `assert(rf_wr_sel == 3);
        `assert(rf_wr_data == expected_quot);
        rf_wr_ack <= 1;

        #10
        `assert(busy);
        `assert(rf_wr_req);
        `assert(rf_wr_sel == 7);
        `assert(rf_wr_data == expected_mod);
        rf_wr_ack <= 1;

        #10
        rf_wr_ack <= 0;
        #10;

        `assert(~rf_wr_req);
        `assert(~busy);
    endtask

    task test_div(
        [data_width - 1:0] _a,
        [data_width - 1:0] _b,
        [data_width - 1:0] expected_quot,
        [data_width - 1:0] expected_mod
    );
        `assert(~busy);
        `assert(~rf_wr_ack);

        $display("");
        $display("==================");
        $display("test %0d / %0d => %0d r %0d", _a, _b, expected_quot, expected_mod);
        a <= _a;
        b <= _b;  // 81 r 37
        req <= 1;
        cnt = 0;
        r_quot_sel <= 3;
        r_mod_sel <= 7;

        #10
        req <= 0;
        a <= 0;
        b <= 0;
        r_quot_sel <= 0;
        r_mod_sel <= 0;
        `assert(busy);
        `assert(~rf_wr_req);

        while(~rf_wr_req) begin
            `assert(busy);
            cnt = cnt + 1;
            #10;
        end
        $display("after cnt loop %0d", cnt);
        `assert (cnt == 32);

        `assert(rf_wr_req);
        `assert(busy);
        $display("rf_wr_data %0d expected_quot %0d", rf_wr_data, expected_quot);
        `assert (rf_wr_data == expected_quot);
        `assert(rf_wr_sel == 3);

        #10
        `assert(rf_wr_req);
        `assert(busy);
        $display("rf_wr_data %0d expected_quot %0d", rf_wr_data, expected_quot);
        `assert (rf_wr_data == expected_quot);
        `assert(rf_wr_sel == 3);

        #10 // posedge clk
        `assert(rf_wr_req);
        `assert(busy);
        $display("rf_wr_data %0d expected_quot %0d", rf_wr_data, expected_quot);
        `assert (rf_wr_data == expected_quot);
        `assert(rf_wr_sel == 3);
        rf_wr_ack = 1;
        // #5
        
        #10 // posedge clk
        `assert(rf_wr_req);
        `assert(busy);
        `assert(rf_wr_sel == 7);
        $display("rf_wr_data %0d expected_mod %0d", rf_wr_data, expected_mod);
        `assert (rf_wr_data == expected_mod);
        #10

        rf_wr_ack <= 1;
        `assert(~rf_wr_req);
        `assert(~busy);

        #10
        rf_wr_ack <= 0;

        #10;
        assert(~rf_wr_ack);
        assert(~req);
    endtask

    initial begin
        $monitor("t=%0d a=%0d b=%0d rf_wr_data=%0d rf_wr_sel=%0d busy=%0b rf_wr_req=%0b", $time, a, b, rf_wr_data, rf_wr_sel, busy, rf_wr_req);
        rst = 1;

        #10;
        #5;
        rst = 0;
        $display("reset going low");
        rf_wr_ack = 0;
        req = 0;
        `assert(~busy);
        `assert(~rf_wr_req);

        #10;
        `assert(~busy);
        `assert(~rf_wr_req);

        #10;
        `assert(~busy);
        `assert(~rf_wr_req);

        #10

        test_div(10000, 123, 81, 37);
        test_div_zero(0, 0, '1, 0);
        test_div_zero(2, 0, '1, 2);
        test_div(4, 4, 1, 0);
        test_div(4, 2, 2, 0);
        test_div(0, 2, 0, 0);
        test_div(4294967295, 1234567, 3478, 1143269);
        test_div(4294967295, 4294967295, 1, 0);
        test_div(4294967295, 4294967293, 1, 2);

        #10 $finish;
    end
endmodule
