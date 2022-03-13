/*
take in various request, selector and data lines, to write to registry file
pick one, and write it, and ack it. others will have to stall

since yosys doesnt accept memories as ports, we create multiple versions, for 
varying numbers of input ports
*/
module proc_reg_wr_mux2(
    input clk,
    // input rst,

    input [reg_sel_width - 1:0] reg_sel0,
    input [data_width - 1:0] data0,
    input req0,
    output reg ack0,

    input [reg_sel_width - 1:0] reg_sel1,
    input [data_width - 1:0] data1,
    input req1,
    output reg ack1,

    // input [reg_sel_width - 1:0] reg_sel2,
    // input [data_width - 1:0] data2,
    // input req2,
    // output reg ack2,

    // input [reg_sel_width - 1:0] reg_sel3,
    // input [data_width - 1:0] data3,
    // input req3,
    // output reg ack3,

    output reg [reg_sel_width - 1:0] rf_wr_reg_sel,
    output reg [data_width - 1:0] rf_wr_data,
    output reg rf_wr_req
);
    // parameter num_ports = 3;

    always @(posedge clk) begin
        // if(rst) begin
        //     rf_wr_req <= 0;
        //     ack0 <= 0;
        //     ack1 <= 0;
        //     ack2 <= 0;
        //     ack3 <= 0;
        // end else begin
            rf_wr_req <= 0;
            ack0 <= 0;
            ack1 <= 0;
            // ack2 <= 0;
            // ack3 <= 0;

            if (req0) begin
                ack0 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel0;
                rf_wr_data <= data0;
            end else if(req1) begin
                ack1 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel1;
                rf_wr_data <= data1;
            // end else if(req2) begin
            //     ack2 <= 1;
            //     rf_wr_req <= 1;
            //     rf_wr_reg_sel <= reg_sel2;
            //     rf_wr_data <= data2;
            // end else if(req3) begin
            //     ack3 <= 1;
            //     rf_wr_req <= 1;
            //     rf_wr_reg_sel <= reg_sel3;
            //     rf_wr_data <= data3;
            end
        // end
    end
endmodule

module proc_reg_wr_mux3(
    input clk,
    // input rst,

    input [reg_sel_width - 1:0] reg_sel0,
    input [data_width - 1:0] data0,
    input req0,
    output reg ack0,

    input [reg_sel_width - 1:0] reg_sel1,
    input [data_width - 1:0] data1,
    input req1,
    output reg ack1,

    input [reg_sel_width - 1:0] reg_sel2,
    input [data_width - 1:0] data2,
    input req2,
    output reg ack2,

    // input [reg_sel_width - 1:0] reg_sel3,
    // input [data_width - 1:0] data3,
    // input req3,
    // output reg ack3,

    output reg [reg_sel_width - 1:0] rf_wr_reg_sel,
    output reg [data_width - 1:0] rf_wr_data,
    output reg rf_wr_req
);
    // parameter num_ports = 3;

    always @(posedge clk) begin
        // if(rst) begin
        //     rf_wr_req <= 0;
        //     ack0 <= 0;
        //     ack1 <= 0;
        //     ack2 <= 0;
        //     ack3 <= 0;
        // end else begin
            rf_wr_req <= 0;
            ack0 <= 0;
            ack1 <= 0;
            ack2 <= 0;
            // ack3 <= 0;

            if (req0) begin
                ack0 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel0;
                rf_wr_data <= data0;
            end else if(req1) begin
                ack1 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel1;
                rf_wr_data <= data1;
            end else if(req2) begin
                ack2 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel2;
                rf_wr_data <= data2;
            // end else if(req3) begin
            //     ack3 <= 1;
            //     rf_wr_req <= 1;
            //     rf_wr_reg_sel <= reg_sel3;
            //     rf_wr_data <= data3;
            end
        // end
    end
endmodule

module proc_reg_wr_mux4(
    input clk,
    // input rst,

    input [reg_sel_width - 1:0] reg_sel0,
    input [data_width - 1:0] data0,
    input req0,
    output reg ack0,

    input [reg_sel_width - 1:0] reg_sel1,
    input [data_width - 1:0] data1,
    input req1,
    output reg ack1,

    input [reg_sel_width - 1:0] reg_sel2,
    input [data_width - 1:0] data2,
    input req2,
    output reg ack2,

    input [reg_sel_width - 1:0] reg_sel3,
    input [data_width - 1:0] data3,
    input req3,
    output reg ack3,

    output reg [reg_sel_width - 1:0] rf_wr_reg_sel,
    output reg [data_width - 1:0] rf_wr_data,
    output reg rf_wr_req
);
    // parameter num_ports = 3;

    always @(posedge clk) begin
        // if(rst) begin
        //     rf_wr_req <= 0;
        //     ack0 <= 0;
        //     ack1 <= 0;
        //     ack2 <= 0;
        //     ack3 <= 0;
        // end else begin
            rf_wr_req <= 0;
            ack0 <= 0;
            ack1 <= 0;
            ack2 <= 0;
            ack3 <= 0;

            if (req0) begin
                ack0 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel0;
                rf_wr_data <= data0;
            end else if(req1) begin
                ack1 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel1;
                rf_wr_data <= data1;
            end else if(req2) begin
                ack2 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel2;
                rf_wr_data <= data2;
            end else if(req3) begin
                ack3 <= 1;
                rf_wr_req <= 1;
                rf_wr_reg_sel <= reg_sel3;
                rf_wr_data <= data3;
            end
        // end
    end
endmodule
