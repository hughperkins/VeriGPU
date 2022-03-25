// try measuring timing...

function add_one(input [31:0] pc, output [31:0] pc2);
    pc2 = pc + 1;
endfunction

module add_one_m(input[31:0] pc, output [31:0] pc2);
    assign pc2 = pc + 1;
endmodule
