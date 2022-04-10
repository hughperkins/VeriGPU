// represents GPU global memory
// we add in simulated delay

// `timescale 1ns/10ps

module global_mem_controller (
    input clk,
    input rst,
    // input ena,  // enables incoming requests to be processed. whilst this is low, incoming requests are stored
                // (only a single request can be stored), and once this goes high, it will be processed
                // this lets us turn off reset, load in our program into memory, then turn on enable
                // and the processor starts running

    input core1_rd_req,
    input core1_wr_req,

    input [addr_width - 1:0]      core1_addr,
    output reg [data_width - 1:0] core1_rd_data,
    input [data_width - 1:0]      core1_wr_data,

    output reg core1_busy,
    output reg core1_ack,

    // for use by comp_driver.sv; might migrate to use contr_ in the future, perhaps
    // no simulated delay added
    /*
    input                    oob_wr_en,
    input [addr_width - 1:0] oob_wr_addr,
    input [data_width - 1:0] oob_wr_data,
    */

    // for use by controller.sv
    // we'll probalby add siulated delay to this
    input                    contr_wr_en,
    input                    contr_rd_en,
    input [addr_width - 1:0] contr_wr_addr,
    input [data_width - 1:0] contr_wr_data,
    input [addr_width - 1:0] contr_rd_addr,
    output reg [data_width - 1:0] contr_rd_data,
    output reg contr_rd_ack
);
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

    // reg n_contr_rd_ack;

    reg [data_width - 1:0] n_rd_data;

    always @(*) begin
    // $monitor("t=%0d mem.always*.mon rst=%0d ena=%0d rd_req=%0d wr_req=%0d addr=%0d rd_data=%0d wr_data=%0d busy=%0d ack=%0d clks_to_wait=%0d",
    //   $time, rst, ena, rd_req, wr_req, addr, rd_data, wr_data, busy, ack, clks_to_wait);
    // $display("t=%0d mem.always*.disp rst=%0d ena=%0d rd_req=%0d wr_req=%0d addr=%0d rd_data=%0d wr_data=%0d busy=%0d ack=%0d clks_to_wait=%0d",
    //   $time, rst, ena, rd_req, wr_req, addr, rd_data, wr_data, busy, ack, clks_to_wait);
    // $display("t=%0d mem.always*.strb rst=%0d ena=%0d rd_req=%0d wr_req=%0d addr=%0d rd_data=%0d wr_data=%0d busy=%0d ack=%0d clks_to_wait=%0d",
    //   $time, rst, ena, rd_req, wr_req, addr, rd_data, wr_data, busy, ack, clks_to_wait);

        n_ack = 0;
        n_busy = 0;

        n_received_rd_req = received_rd_req;
        n_received_wr_req = received_wr_req;

        n_rd_data = '0;
        n_received_addr = received_addr;
        n_received_data = received_data;

        n_write_now = 0;
        n_read_now = 0;

        n_clks_to_wait = 0;

        // n_contr_rd_ack = 0;

        // $display("rst %0d received_rd_req=%0d", rst, received_rd_req);
        `assert_known(received_rd_req);
        `assert_known(received_wr_req);
        `assert_known(core1_wr_req);
        `assert_known(core1_rd_req);
        // `assert_known(ena);
        if (received_rd_req) begin
            `assert_known(clks_to_wait);
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
            `assert_known(clks_to_wait);
            if (clks_to_wait == 0) begin
                n_ack = 1;
                n_write_now = 1;
                n_received_rd_req = 0;
                n_received_wr_req = 0;
                n_busy = 0;
            end else begin
                n_clks_to_wait = clks_to_wait - 1;
                n_busy = 1;
            end
        end else if (core1_wr_req) begin
            n_received_wr_req = 1;
            n_clks_to_wait = mem_simulated_delay - 1;
            // $display("writing addr=%0d", addr);
            n_received_addr = core1_addr;
            n_received_data = core1_wr_data;
            n_ack = 0;
            n_busy = 1;
        end else if (core1_rd_req) begin
            n_received_rd_req = 1;
            n_clks_to_wait = mem_simulated_delay - 1;
            // $display("reading addr=%0d", addr);
            n_received_addr = core1_addr;
            n_ack = 0;
            n_busy = 1;
        end
    end

    always @(posedge clk, negedge rst) begin
        `assert_known(rst);
        if(~rst) begin
            $display("mem_delayed.rst");
            clks_to_wait <= 0;
            core1_busy <= 0;
            core1_ack <= 0;
            core1_rd_data <= '0;

            received_addr <= 0;
            received_data <= 0;

            received_rd_req <= 0;
            received_wr_req <= 0;

            contr_rd_ack <= 0;
        end else begin
            // $display("mem_delayed.clk non reset");
            /*
            `assert_known(oob_wr_en);
            if(oob_wr_en) begin
                // $display("oob_wen mem[%0d] = %0d", oob_wr_addr, oob_wr_data);
                mem[oob_wr_addr >> 2] <= oob_wr_data;
            end
            */

            contr_rd_ack <= 0;

            if(contr_wr_en) begin
                // $display("mem controller contr wr en writing %0d to addr %0d", contr_wr_data, contr_wr_addr);
                mem[contr_wr_addr >> 2] <= contr_wr_data;
            end

            if(contr_rd_en) begin
                // $display("mem controller contr rd en reading %0d from addr %0d", mem[contr_rd_addr >> 2], contr_rd_addr);
                contr_rd_data <= mem[contr_rd_addr >> 2];
                contr_rd_ack <= 1;
            end

            // if(ena) begin
            //     $display(
            //         "t=%0d mem_delayed.ff n_clks=%0d n_received_rd_req=%0d n_received_wr_req=%0d n_ack=%0d n_busy=%0d n_received_addr=%0d n_read_now=%0d mem[n_received_addr]=%0d",
            //         $time, n_clks_to_wait, n_received_rd_req, n_received_wr_req, n_ack, n_busy, n_received_addr, n_read_now, mem[n_received_addr]);
            // end
            clks_to_wait <= n_clks_to_wait;
            core1_busy <= n_busy;
            core1_ack <= n_ack;
            core1_rd_data <= '0;

            received_addr <= n_received_addr;
            received_data <= n_received_data;

            received_rd_req <= n_received_rd_req;
            received_wr_req <= n_received_wr_req;

            `assert_known(n_write_now);
            if(n_write_now) begin
                // $display("writing now n_received_data=%0d n_received_addr=%0d", n_received_data, n_received_addr);
                mem[{2'b0, n_received_addr[31:2]}] <= n_received_data;
            end

            `assert_known(n_read_now);
            if(n_read_now) begin
                // $display(
                //     "reading rd data n_received_addr=%0d mem[ {2'b0, n_received_addr[31:2]} ]=%0d",
                //     n_received_addr, mem[ {2'b0, n_received_addr[31:2]} ]);
                core1_rd_data <= mem[ {2'b0, n_received_addr[31:2]} ];
            end
        end
    end
endmodule
