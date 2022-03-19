`timescale 1ns/10ps

module mem_delayed_test();
    reg clk;
    wire [data_width - 1:0] rd_data;
    reg [data_width - 1:0] wr_data;
    reg [addr_width - 1:0] addr;
    wire ack;
    wire busy;

    reg rd_req;
    reg wr_req;
    reg rst;

    reg oob_wen;

    mem_delayed mem_delayed1 (
        .clk(clk),
        .rst(rst),
        .busy(busy),
        .ack(ack),
        .rd_req(rd_req),
        .wr_req(wr_req),
        .rd_data(rd_data),
        .wr_data(wr_data),
        .addr(addr),
        .oob_wen(oob_wen)
    );

    task check_read(
        [addr_width - 1:0] tgt_addr,
        [data_width - 1:0] expected_data,
        [7:0] expected_cycles
    );
        reg [7:0] cycles;
        $display("===========");
        $display("test read from %0d", tgt_addr);
        $display("check read addr=%0d exp_data=%0d", tgt_addr, expected_data);
        cycles = 0;
        `assert (~busy);
        addr <= tgt_addr;
        wr_req <= 1'b0;
        rd_req <= 1'b1;
        #5;
        #5;
        `assert(busy);
        do begin
            `assert(busy);
            cycles = cycles + 1;
            #10;
        end while(~ack && cycles < 1000);
        `assert(~busy);
        `assert(rd_data == expected_data);
        `assert (cycles == expected_cycles);
    endtask

    task write(
        [addr_width - 1:0] tgt_addr,
        [data_width - 1:0] tgt_data,
        [7:0] expected_cycles
    );
        reg [7:0] cycles;
        $display("===========");
        $display("test write %0d to %0d", tgt_data, tgt_addr);
        cycles <= 0;
        `assert (!busy);
        addr <= tgt_addr;
        wr_data <= tgt_data;
        wr_req <= 1'b1;
        rd_req <= 1'b0;

        #5;
        #5;
        $display("busy=%0d wr_req=%0b", busy, wr_req);
        `assert(busy);
        addr <= '0;
        wr_data <= '0;
        wr_req <= 1'b0;
        do begin
            `assert(busy);
            cycles <= cycles + 1;
            #10;
        end while(~ack && cycles < 1000);
        $display("cycles %0d", cycles);

        `assert(~busy);
        $display("cycles=%0d expected_cycles=%0d", cycles, expected_cycles);
        `assert (cycles == expected_cycles);
    endtask

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%0d test.mon ack=%d busy=%d rd_req=%h wr_req=%h addr=%0h rd_data=%0h wr_data=%0h", $time, ack, busy, rd_req, wr_req, addr, rd_data, wr_data);
        rst = 1;
        rst <= 1;
        wr_req <= 0;
        rd_req <= 0;
        oob_wen <= 0;

        #5
        #10
        rst = 0;
        $display("reset off");
        #10
        $monitor("t=%0d test.mon ack=%d busy=%d rd_req=%h wr_req=%h addr=%0h rd_data=%0h wr_data=%0h", $time, ack, busy, rd_req, wr_req, addr, rd_data, wr_data);

        write(16'd8, 16'd111, mem_simulated_delay);
        write(16'd16, 16'd222, mem_simulated_delay);

        check_read(16'd16, 16'd222, mem_simulated_delay);
        check_read(16'd8, 16'd111, mem_simulated_delay);
        check_read(16'd16, 16'd222, mem_simulated_delay);
        check_read(16'd8, 16'd111, mem_simulated_delay);

        write(16'd20, 16'h333, mem_simulated_delay);
        write(16'd26, 16'h444, mem_simulated_delay);
        check_read(16'd20, 16'h333, mem_simulated_delay);
        check_read(16'd26, 16'h444, mem_simulated_delay);

        check_read(16'd16, 16'd222, mem_simulated_delay);
        check_read(16'd8, 16'd111, mem_simulated_delay);

        write(16'd8, 16'd111, mem_simulated_delay);
        write(16'd11, 16'd222, mem_simulated_delay);
        check_read(16'd8, 16'd222, mem_simulated_delay);

        write(16'd8, 16'd111, mem_simulated_delay);
        write(16'd12, 16'd222, mem_simulated_delay);
        check_read(16'd8, 16'd111, mem_simulated_delay);
        check_read(16'd12, 16'd222, mem_simulated_delay);

        assert(~busy);

        #200 $finish();
    end
endmodule
