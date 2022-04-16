// test running timing.py on hierarchy

module mod1(input clk, input rst, input a, input b, output c);
    always @(posedge clk, negedge rst) begin
        if(~rst) begin
        end else begin
            c <= a & b;
        end
    end
endmodule

module mod2(input clk, input rst, input a, input b, output c);
    mod1 dut(.clk(clk), .rst(rst), .a(a), .b(b), .c(c));
endmodule

module mod3(input clk, input rst, input a, input b, output c);
    mod2 dut(.clk(clk), .rst(rst), .a(a), .b(b), .c(c));
endmodule
