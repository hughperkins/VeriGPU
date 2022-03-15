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

    initial begin
        $monitor("t=%0d a=%0d b=%0d rf_wr_data=%0d rf_wr_sel=%0d busy=%0b rf_wr_req=%0b", $time, a, b, rf_wr_data, rf_wr_sel, busy, rf_wr_req);
        rst = 1;

        #10
        rst = 0;
        rf_wr_ack = 0;
        assert(~busy);
        assert(~rf_wr_req);

        a = 10000;
        b = 123;  // 81 r 37
        req = 1;
        cnt = 0;
        r_quot_sel = 3;
        r_mod_sel = 7;

        #10
        req = 0;
        a = 0;
        b = 0;
        r_quot_sel = 0;
        r_mod_sel = 0;
        assert(busy);
        assert(~rf_wr_req);

        while(~rf_wr_req) begin
            assert(busy);
            cnt = cnt + 1;
            #10;
        end
        $display("after cnt loop %0d", cnt);
        assert (cnt == 32);

        assert(rf_wr_req);
        assert(busy);
        assert (rf_wr_data == 81);
        assert(rf_wr_sel == 3);

        #10
        assert(rf_wr_req);
        assert(busy);
        assert (rf_wr_data == 81);
        assert(rf_wr_sel == 3);

        #10 // posedge clk
        assert(rf_wr_req);
        assert(busy);
        assert (rf_wr_data == 81);
        assert(rf_wr_sel == 3);
        rf_wr_ack = 1;
        // #5
        
        #10 // posedge clk
        assert(rf_wr_req);
        assert(busy);
        assert (rf_wr_data == 37);
        assert(rf_wr_sel == 7);
        #10

        rf_wr_ack = 1;
        assert(~rf_wr_req);
        assert(~busy);

        #10
        rf_wr_ack = 0;

        #10 $finish;
    end
endmodule
