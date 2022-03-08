module test_add_one_chunked();
    reg [31:0] a;
    reg [31:0] out;

    add_one_chunked dut(.pc(a), .pc2(out));

    initial begin
        $monitor("a=%0d out=%0d", a, out);
        a = 15;
        #1
        a = 13;
        #1
        a = 31;
        #1
        a = 63;
        #1
        a = 64;
        #1
        a = 1000;
        #1
        a = 10000;
        #1
        a = 100000;
        #1
        a = 1000000;
        #1
        a = 10000000;
        #1
        a = 100000000;
        #1
        a = 1000000000;

    end
endmodule
