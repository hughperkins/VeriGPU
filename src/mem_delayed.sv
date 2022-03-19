module mem_delayed
    #(parameter mem_simulated_delay=5) (
    input clk,
    input rst,

    input rd_req,
    input wr_req,

    input [addr_width - 1:0]      addr,
    output reg [data_width - 1:0] rd_data,
    input [data_width - 1:0]      wr_data,

    output reg busy,
    output reg ack,

    input                    oob_wen,
    input [addr_width - 1:0] oob_wr_addr,
    input [data_width - 1:0] oob_wr_data
);
    // parameter mem_simulated_delay = 128;

    reg [data_width - 1:0] mem[memory_size];

    reg [addr_width - 1:0] received_addr;
    reg [data_width - 1:0] received_data;
    reg                    received_rd_req;
    reg                    received_wr_req;

    reg [7:0]              clks_to_wait;

    reg                    n_busy;
    reg                    n_ack;

    reg [addr_width - 1:0] n_received_addr;
    reg [data_width - 1:0] n_received_data;
    reg                    n_received_rd_req;
    reg                    n_received_wr_req;

    reg [7:0]              n_clks_to_wait;

    reg                    n_read_now;
    reg                    n_write_now;

    reg [data_width - 1:0] n_rd_data;

    always @(*) begin
        n_ack = 0;
        n_busy = 0;

        n_received_rd_req = received_rd_req;
        n_received_wr_req = received_wr_req;

        n_rd_data = '0;
        n_received_addr = received_addr;
        n_received_data = received_data;

        n_write_now = 0;
        n_read_now = 0;

        assert(~$isunknown(oob_wen));

        if(~rst) begin
            assert(~$isunknown(received_rd_req));
            assert(~$isunknown(received_wr_req));
            assert(~$isunknown(wr_req));
            assert(~$isunknown(rd_req));
        end
        if (received_rd_req) begin
            assert(~$isunknown(clks_to_wait));
            if (clks_to_wait == 0) begin
                n_ack = 1;
                n_read_now = 1;
                // n_rd_data <= mem[{2'b0, received_addr[31:2]}];
                n_received_rd_req = 0;
                n_received_wr_req = 0;
                n_busy = 0;
            end else begin
                n_clks_to_wait = clks_to_wait - 1;
                n_busy = 1;
            end
        end else if(received_wr_req) begin
            assert(~$isunknown(clks_to_wait));
            if (clks_to_wait == 0) begin
                n_ack = 1;
                n_write_now = 1;
                n_received_rd_req <= 0;
                n_received_wr_req <= 0;
                n_busy = 0;
            end else begin
                n_clks_to_wait = clks_to_wait - 1;
                n_busy = 1;
            end
        end else if (wr_req) begin
            n_received_wr_req = 1;
            n_clks_to_wait = mem_simulated_delay - 1;
            n_received_addr = addr;
            n_received_data = wr_data;
            n_ack = 0;
            n_busy = 1;
        end else if (rd_req) begin
            n_received_rd_req = 1;
            n_clks_to_wait = mem_simulated_delay - 1;
            n_received_addr = addr;
            n_ack = 0;
            n_busy = 1;
        end
    end

    always @(posedge clk, posedge rst) begin
        assert(~$isunknown(rst));
        if(rst) begin
            clks_to_wait <= 0;
            busy <= 0;
            ack <= 0;
            rd_data <= '0;

            received_addr <= 0;
            received_data <= 0;

            received_rd_req <= 0;
            received_wr_req <= 0;

            if(oob_wen) begin
                mem[oob_wr_addr] <= oob_wr_data;
            end
        end else begin
            $display(
                "t=%0d mem_delayed.ff n_clks=%0d n_received_rd_req=%0d n_received_wr_req=%0d n_ack=%0d n_busy=%0d n_received_addr=%0d n_read_now=%0d mem[n_received_addr]=%0d",
                $time, n_clks_to_wait, n_received_rd_req, n_received_wr_req, n_ack, n_busy, n_received_addr, n_read_now, mem[n_received_addr]);
            clks_to_wait <= n_clks_to_wait;
            busy <= n_busy;
            ack <= n_ack;
            rd_data <= '0;

            received_addr <= n_received_addr;
            received_data <= n_received_data;

            received_rd_req <= n_received_rd_req;
            received_wr_req <= n_received_wr_req;

            if(n_write_now) begin
                mem[{2'b0, n_received_addr[31:2]}] <= n_received_data;
            end

            if(n_read_now) begin
                rd_data <= mem[ {2'b0, n_received_addr[31:2]} ];
            end
        end
    end
endmodule
