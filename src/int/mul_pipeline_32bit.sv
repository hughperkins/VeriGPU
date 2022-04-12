/*

width=32 bits_per_cycle=2
Max propagation delay: 59.4 nand units
Area:                  2283.0 nand units

*/
module mul_pipeline_32bit(
    input clk,
    input rst,
    input req,
    input [width - 1:0] a,
    input [width - 1:0] b,
    output reg [width - 1:0] out,
    output reg ack
);
    parameter width = 32;
    parameter bits_per_cycle = 2;

    reg [width - 1:0] internal_a;
    reg [width - 1:0] internal_b;
    reg [$clog2(width):0] cin;
    reg [$clog2(width):0] pos;

    reg [1:0] state;

    reg [$clog2(width):0] cout;

    reg [width - 1:0] n_out;
    reg n_ack;

    reg [width - 1:0] n_internal_a;
    reg [width - 1:0] n_internal_b;
    reg [$clog2(width):0] n_cin;
    reg [$clog2(width):0] n_pos;

    reg [1:0] n_state;

    typedef enum bit [1:0] {
        IDLE,
        MUL1
        // OUT
    } e_state;

    // parameter clog2_bits_per_cycle = $clog2(bits_per_cycle);

    always @(req, state, pos) begin
        // `assert(bits_per_cycle == (1 << clog2_bits_per_cycle));
        // $display("mul @(*) req=%b state=%0d pos=%0d", req, state, pos);

        n_out = out;
        n_ack = 0;

        n_internal_a = internal_a;
        n_internal_b = internal_b;
        n_cin = '0;
        n_pos = pos;

        n_state = state;

        cout = '0;

        case (state)
            IDLE: begin
                `assert_known(req);
                if(req) begin
                    // $display("a %b %0d", a, a);
                    // $display("b %b %0d", b, b);
                    n_internal_a = a;
                    n_internal_b = b;
                    n_pos = 0;
                    n_state = MUL1;
                    n_cin = '0;
                end
            end
            MUL1: begin
                `assert_known(pos);
                `assert_known(internal_a);
                `assert_known(internal_b);
                `assert_known(cin);
                mul_pipeline_cycle_32bit_2bpc(
                    pos,
                    internal_a,
                    internal_b,
                    cin,
                    n_out[pos + bits_per_cycle - 1 -: bits_per_cycle],
                    cout
                );
                n_cin = cout;
                n_pos = pos + bits_per_cycle;
                `assert_known(pos);
                // $display("driver clocked pos=%0d n_out=%b %0d cin=%b n_cin=%b", pos, n_out, n_out, cin, n_cin);
                if (n_pos >= width) begin
                    // $display("final n_out %b %0d", n_out, n_out);
                    n_ack = 1;
                    n_state = IDLE;
                end
            end
            // OUT: begin
            //     n_ack <= 1;
            // end
            default: begin
            end
        endcase
    end

    always @(posedge clk, negedge rst) begin
        if(~rst) begin
            state <= IDLE;
            cin <= '0;
            pos <= '0;
            ack <= 0;
            out <= '0;
        end else begin
            state <= n_state;
            pos <= n_pos;
            ack <= n_ack;
            cin <= n_cin;
            internal_a <= n_internal_a;
            internal_b <= n_internal_b;
            out <= n_out;
        end
    end
endmodule
