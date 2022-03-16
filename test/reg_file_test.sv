module reg_file_test();
    reg [reg_sel_width - 1:0] rs1_sel;
    reg [reg_sel_width - 1:0] rs2_sel;
    reg [data_width - 1:0] rs1_data;
    reg [data_width - 1:0] rs2_data;
    reg [reg_sel_width - 1:0] wr_sel;
    reg wr_req;
    reg [data_width - 1:0] wr_data;

    reg clk;
    reg rst;

    reg_file reg_file_(
        .rs1_sel(rs1_sel),
        .rs2_sel(rs2_sel),
        .rs1_data(rs1_data),
        .rs2_data(rs2_data),
        .wr_sel(wr_sel),
        .wr_req(wr_req),
        .wr_data(wr_data),

        .clk(clk),
        .rst(rst)
    );

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%0d rst=%0b rs1_data=%0d rs2_data=%0d", $time, rst, rs1_data, rs2_data);

        rst = 1;

        #10
        rst = 0;
        wr_sel = 3;
        wr_data = 111;
        wr_req = 1;
        rs1_sel = 3;
        rs2_sel = 3;

        #10
        assert(rs1_data == 111);
        assert (rs2_data == 111);
        wr_req = 0;
        wr_data = 222;

        #10
        assert(rs1_data == 111);
        assert (rs2_data == 111);
        wr_sel = 7;

        #10
        assert(rs1_data == 111);
        assert (rs2_data == 111);
        wr_req = 1;
        rs2_sel = 7;

        #10
        assert(rs1_data == 111);
        assert (rs2_data == 222);

        #100
        $finish;

    end
endmodule
