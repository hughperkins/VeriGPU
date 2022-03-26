// parameter width = 32;
// parameter bits_per_cycle = 2;

module mul(
    input [width - 1:0] a,
    input [width - 1:0] b,
    output reg [width - 1:0] out
);
    reg [$clog2(width):0] cin;
    reg [$clog2(width):0] cout;
    always @(*) begin
        cin = '0;
        for(int i = 0; i < width; i += bits_per_cycle) begin
            mul_partial_add_task(
                i,
                a,
                b,
                cin,
                out[i + bits_per_cycle - 1 -: bits_per_cycle],
                cout
            );
            // $display("driver i=%0d out %b", i, out);
            cin = cout;
        end
        // $display("driver after for out %b", out);
    end
endmodule
