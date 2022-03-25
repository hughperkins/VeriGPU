// prototype writing a module around prot_task
module prot_task_module(
    input [1:0] a,
    input [1:0] b,
    output reg [1:0] out
);
    always @(*) begin
        prot_task(a, b, out);
    end
endmodule
