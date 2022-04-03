// this is going to try to simulate what proc does approximately
module call_div(
    output reg [$clog2(data_width) - 1:0] div_pos,
    input clk,
    input rst,
    input trigger_div,
    input [data_width - 1:0] div_a,
    input [data_width - 1:0] div_b,
    output reg div_busy
);
    reg div_req;
    // reg div_busy
    int_div_regfile dut(
        .clk(clk),
        .rst(rst),
        .req(div_req),
        .busy(div_busy),
        // .r_quot_sel(r_quot_sel),
        // .r_mod_sel(r_mod_sel),
        .a(div_a),
        .b(div_b),
        // .rf_wr_sel(rf_wr_sel),
        // .rf_wr_data(rf_wr_data),
        // .rf_wr_req(rf_wr_req),
        // .rf_wr_ack(rf_wr_ack)

        .pos(div_pos)
    );

    always @(*) begin
        
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            div_req <= 0;
        end else begin
            if (trigger_div) begin
                div_req <= 1;
            end else begin
                div_req <= 0;
            end
        end
    end
endmodule

module int_div_regfile_test();
    reg clk;
    reg rst;

    reg trigger_div;
    reg div_busy;

    reg [reg_sel_width - 1: 0] r_quot_sel;  // 0 means, dont write (i.e. x0)
    reg [reg_sel_width - 1: 0] r_mod_sel;   // 0 means, dont write  (i.e. x0)
    reg [data_width - 1:0] div_a;
    reg [data_width - 1:0] div_b;

    reg [reg_sel_width - 1:0] rf_wr_sel;
    reg [data_width - 1:0] rf_wr_data;
    reg rf_wr_req;
    reg rf_wr_ack;

    reg [31:0] cnt;

    reg [$clog2(data_width) - 1:0] div_pos;

    call_div call_div_(
        .clk(clk),
        .rst(rst),
        .trigger_div(trigger_div),
        .div_a(div_a),
        .div_b(div_b),
        .div_pos(div_pos),
        .div_busy(div_busy)
    );

    task up();
        $display("  +");
        #5 clk = 1;
    endtask

    task down();
        $display("-");
        #5 clk = 0;
    endtask

    task tick();
        $display("-");
        #5 clk = 0;
        $display("  +");
        #5 clk = 1;
    endtask

    initial begin
        $display("test2");
        $monitor("t=%0d trig=%0d busy=%0d pos=%0d", $time, trigger_div, div_busy, div_pos);
        rst = 0;
        trigger_div = 0;
        tick();

        rst = 1;
        tick();

        trigger_div = 1;
        tick();
        tick();
        tick();

        #10 $finish;
    end
endmodule
