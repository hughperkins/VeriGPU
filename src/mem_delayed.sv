module mem_delayed(
    input clk,  rst, input wr_req, input rd_req,
    output reg busy, output reg ack,
    input [31:0] addr,
    output reg [31:0] rd_data,
    input [31:0] wr_data,

    input [31:0] oob_wr_addr,
    input [31:0] oob_wr_data,
    input oob_wen
);
    reg [31:0] mem[256];
    reg [31:0] received_addr;
    reg [31:0] received_data;
    reg received_rd_req;
    reg received_wr_req;
    reg [4:0] clks_to_wait;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            received_rd_req <= 0;
            received_wr_req <= 0;
            busy <= 0;
        end else if(oob_wen) begin
            mem[oob_wr_addr] <= oob_wr_data;
        end else begin
            if (received_rd_req) begin
                if (clks_to_wait == 0) begin
                    ack <= 1;
                    rd_data <= mem[{2'b0, received_addr[31:2]}];
                    received_rd_req <= 0;
                    received_wr_req <= 0;
                    busy <= 0;
                end else begin
                    clks_to_wait <= clks_to_wait - 1;
                end
            end else if(received_wr_req) begin
                if (clks_to_wait == 0) begin
                    ack <= 1;
                    mem[{2'b0, received_addr[31:2]}] <= received_data;
                    received_rd_req <= 0;
                    received_wr_req <= 0;
                    busy <= 0;
                end else begin
                    clks_to_wait <= clks_to_wait - 1;
                end
            end else if (wr_req) begin
                received_wr_req <= 1;
                clks_to_wait <= 4;
                received_addr <= addr;
                received_data <= wr_data;
                ack <= 0;
                busy <= 1;
            end else if (rd_req) begin
                received_rd_req <= 1;
                clks_to_wait <= 4;
                received_addr <= addr;
                ack <= 0;
                busy <= 1;
            end else begin
                ack <= 0;
                busy <= 0;
            end
        end
    end
endmodule
