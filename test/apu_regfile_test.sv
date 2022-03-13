/*
integeration test of apu and regfile together
*/

module apu_regfile_test();
    reg clk, rst;

    // incoming request
    reg [data_width -1:0] rs1;
    reg [data_width - 1:0] rs2;
    reg [reg_sel_width - 1:0] rd_sel;
    reg [9:0] funct;
    reg busy;
    reg req;

    // connection to reg file
    reg apu_wr_req;
    reg [reg_sel_width -1:0] apu_wr_sel;
    reg [data_width - 1:0] apu_wr_data;

    reg [reg_sel_width - 1:0] proc_rs1_sel;
    reg [reg_sel_width - 1:0] proc_rs2_sel;
    reg [data_width -1:0] proc_rs1_data;
    reg [data_width -1:0] proc_rs2_data;

    apu apu_(
        .clk(clk),
        .rst(rst),
        .rs1(rs1),
        .rs2(rs2),
        .req(req),
        .rd_sel(rd_sel),
        .funct(funct),
        .busy(busy),
        .apu_wr_req(apu_wr_req),
        .apu_wr_sel(apu_wr_sel),
        .apu_wr_data(apu_wr_data)
    );
    reg_file reg_file_(
        .clk(clk), .rst(rst),

        .apu_wr_req(apu_wr_req),
        .apu_ack(apu_ack),
        .apu_wr_sel(apu_wr_sel),
        .apu_wr_data(apu_wr_data),

        .proc_rs1_sel(proc_rs1_sel),
        .proc_rs2_sel(proc_rs2_sel),
        .proc_rs1_data(proc_rs1_data),
        .proc_rs2_data(proc_rs2_data)
    );

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%0d rs1=%0d rs2=%0d busy=%0b apu_wr_req=%0b apu_wr_sel=%0d apu_wr_data=%0d proc_rs1_data=%0d", $time, rs1, rs2, busy, apu_wr_req, apu_wr_sel, apu_wr_data, proc_rs1_data);

        rst = 1;
        #10

        assert (busy == 0);
        assert (apu_wr_req == 0);

        rst = 0;
        rs1 = 1234;
        rs2 = 53;  // q = 23 r 15
        rd_sel = 11;
        funct = DIVU;
        req = 1;

        #10
        req = 0;

        #10
        assert(~busy);
        while(~apu_wr_req) begin
            assert(~busy);
            #10;
        end
        assert(~busy);
        assert(apu_wr_sel == 11);
        assert(apu_wr_data == 23);
        proc_rs1_sel = 11;

        #10
        assert(~apu_wr_req);
        assert(proc_rs1_data == 23);

        rst = 0;
        rs1 = 55555;
        rs2 = 173;  // q = 321 r 22
        rd_sel = 7;
        funct = DIVU;
        req = 1;

        #10
        req = 0;

        #10
        assert(~busy);
        req = 1;

        #10
        assert(busy);
        req = 0;

        #10
        assert(~busy);

        while(~apu_wr_req) begin
            assert(~busy);
            #10;
        end
        assert(~busy);
        assert(apu_wr_sel == 7);
        assert(apu_wr_data == 321);
        proc_rs1_sel = 7;

        #10
        assert(proc_rs1_data == 321);

        $finish;
    end
endmodule
