// purely combinatorial; no pipeline

parameter bitwidth = 32;

module int_div(input [bitwidth - 1:0] a, input [bitwidth - 1:0] b, output reg [bitwidth - 1:0] quotient, output reg [bitwidth - 1:0] remainder);
    reg [bitwidth - 1:0] result1[bitwidth];
    reg [bitwidth - 1:0] result2[bitwidth];

    reg [bitwidth - 1: 0] a_;
    reg [2 * bitwidth - 1: 0] shiftedb;

    always @(*) begin
        quotient = '0;
        a_ = a;
        for(int i = bitwidth - 1; i >= 0; i--) begin
            shiftedb = b << i;
            // $strobe("shiftedb %b quotient %b a_ %b", shiftedb, quotient, a_);
            if (shiftedb < a_) begin
                a_ = a_ - shiftedb;
                quotient[i] = 1;
            end else begin
            end
        end
        remainder = a_;
        $strobe("shiftedb %b", shiftedb);
        $strobe("quotient %b", quotient);
        $strobe("remainder %b", remainder);
        $strobe("a_ %b", a_);
    end
endmodule

/*

1010   # 10
11     #  3
 0
00
101
 11
  1
 11
010
 100
  11
   1
  01

q = 011  # 3
r =   1  # 1

*/
