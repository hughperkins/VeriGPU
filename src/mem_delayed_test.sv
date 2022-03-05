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

    mem_delayed mem_delayed1(
        .clk(clk), .busy(busy), .ack(ack), .rst(rst),
        .rd_req(rd_req), .wr_req(wr_req),
        .rd_data(rd_data), .wr_data(wr_data), .addr(addr)
    );

    task check_read(
        [15:0] tgt_addr,
        [15:0] expected_data,
        [4:0] expected_cycles
    );
        reg [4:0] cycles;
        cycles = 0;
        assert (~busy);
        addr = tgt_addr;
        wr_req = 1'b0;
        rd_req = 1'b1;
        #10
        assert(busy);
        do begin
            assert(~ack);
            cycles = cycles + 1;
            #10;
        end while(busy);
        assert(ack);
        assert(rd_data == expected_data);
        $display("cycles %d", cycles);
        assert (cycles == expected_cycles);
    endtask

    task write(
        [15:0] tgt_addr,
        [15:0] tgt_data,
        [4:0] expected_cycles
    );
        reg [4:0] cycles;
        cycles = 0;
        assert (~busy);
        addr = tgt_addr;
        wr_data = tgt_data;
        wr_req = 1'b1;
        rd_req = 1'b0;
        #10
        assert(busy);
        addr = 'x;
        wr_data = 'x;
        wr_req = 1'b0;
        do begin
            assert(~ack);
            cycles = cycles + 1;
            #10;
        end while(busy);
        assert(ack);
        $display("cycles %d", cycles);
        assert (cycles == expected_cycles);
    endtask

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

        write(16'h8, 16'hab, 5);
        write(16'h10, 16'hcd, 5);

        check_read(16'h10, 16'hcd, 5);
        check_read(16'h8, 16'hab, 5);
        check_read(16'h10, 16'hcd, 5);
        check_read(16'h8, 16'hab, 5);

        write(16'h12, 16'h11, 5);
        write(16'h14, 16'h22, 5);
        check_read(16'h12, 16'h11, 5);
        check_read(16'h14, 16'h22, 5);

        check_read(16'h10, 16'hcd, 5);
        check_read(16'h8, 16'hab, 5);

        #200 $finish();
    end
endmodule
