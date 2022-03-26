// parameter width = 32;
parameter bits_per_cycle = 1;

module mul(
    input [width - 1:0] a,
    input [width - 1:0] b,
    output reg [width - 1:0] out
);
    reg [$clog2(width):0] cin;
    reg [$clog2(width):0] cout;
    always @(*) begin
        cin = '0;
        for(int i = 0; i < width; i++) begin
            mul_partial_add_task(
                i,
                a,
                b,
                cin,
                out[i],
                cout
            );
            cin = cout;
        end
    end
endmodule
