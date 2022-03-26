// parameter width = 32;
// parameter bits_per_cycle = 2;

module mul(
    input [width - 1:0] a,
    input [width - 1:0] b,
    output reg [width - 1:0] out
);
    always @(*) begin
        reg [width * 2 - 1:0] wide_out;
        reg [width * 2 - 1:0] new_out;
        wide_out = '0;
        for(int pos = 0; pos < width; pos += 1) begin
            mul_partial_per_cycle_task(
                pos,
                a,
                b,
                wide_out,
                new_out
            );
            wide_out = new_out;
        end
        out = wide_out[width - 1:0];
    end
endmodule
