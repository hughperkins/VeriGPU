module mem_delayed2(
    input clk,  rst, input wr_req, input rd_req,
    output reg busy, output reg ack,
    input [15:0] addr, output reg [15:0] rd_data,
    input [15:0] wr_data
);
    reg [15:0] mem[256];
    reg [15:0] received_addr;
    reg [15:0] received_data;
    reg received_rd_req;
    reg received_wr_req;
    reg [4:0] clks_to_wait;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            received_rd_req <= 0;
            received_wr_req <= 0;
            busy <= 0;
        end else begin
            if (received_rd_req) begin
                if (clks_to_wait == 0) begin
                    ack <= 1;
                    rd_data <= mem[received_addr];
                    received_rd_req <= 0;
                    received_wr_req <= 0;
                    busy <= 0;
                end else begin
                    clks_to_wait <= clks_to_wait - 1;
                end
            end else if(received_wr_req) begin
                if (clks_to_wait == 0) begin
                    ack <= 1;
                    mem[received_addr] <= received_data;
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
