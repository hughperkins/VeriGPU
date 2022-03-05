module mem_delayed_test();
    reg clk;
    wire [15:0] rd_data;
    reg [15:0] wr_data;
    reg [15:0] addr;
    wire ack;
    wire busy;
    reg rd_req;
    reg wr_req;
    reg rst;

    mem_delayed2 mem_delayed2_(
        .clk(clk), .busy(busy), .ack(ack), .rst(rst),
        .rd_req(rd_req), .wr_req(wr_req),
        .rd_data(rd_data), .wr_data(wr_data), .addr(addr)
    );

    initial begin
        clk = 1;
        forever begin
        #5 clk = ~clk;
        end
    end
    initial begin
        $monitor("t=%d ack=%d busy=%d rd_req=%h wr_req=%h addr=%h rd_data=%h wr_data=%h", $time, ack, busy, rd_req, wr_req, addr, rd_data, wr_data);
        rst = 1;

        #10
        rst = 0;
        addr = 16'h8;
        wr_data = 16'hab;
        wr_req = 1'b1;
        rd_req = 1'b0;

        #10
        wr_req = 1'b0;

        #60
        addr = 16'h10;
        wr_data = 16'hcd;
        wr_req = 1'b1;
        rd_req = 1'b0;

        #10
        wr_req = 1'b0;

        #60
        addr = 16'h8;
        wr_req = 1'b0;
        rd_req = 1'b1;

        #10
        rd_req = 1'b0;

        #60
        addr = 16'h10;
        wr_req = 1'b0;
        rd_req = 1'b1;

        #10
        rd_req = 1'b0;

        #50

        #200 $finish();
    end
endmodule
