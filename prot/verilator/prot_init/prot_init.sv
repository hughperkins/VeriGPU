module prot_init(
    input clk,
    input rst,
    output reg [31:0] out,
    output reg [31:0] state
);
    reg[31:0] n_state;
    always @(state) begin
        n_state = state;
        $display("always out %0d n_state=%0d state=%0d", out, n_state, state);
    end
    always @(posedge clk, negedge rst) begin
        $display("clocked rst %0d out %0d n_state=%0d state=%0d", rst, out, n_state, state);
        if(~rst) begin
            state <= '0;
        end else begin
            // state <= '0;
            state <= n_state;
        end
    end
endmodule
