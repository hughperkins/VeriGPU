// test using timing.py on something with a posedge always
module clocked_counter(input clk, input rst, output reg [31:0] cnt);
    always @(posedge clk, posedge rst) begin
        if(rst) begin
            cnt <= '0;
        end else begin
            cnt <= cnt + 1;
        end
    end
endmodule
