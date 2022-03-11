module test_int_div();
    reg [bitwidth - 1:0] a;
    reg [bitwidth - 1:0] b;
    reg [bitwidth - 1:0] quotient;
    reg [bitwidth - 1:0] remainder;

    int_div dut(.a(a), .b(b), .quotient(quotient), .remainder(remainder) );

    initial begin
        $monitor("t=%0d a=%0d b=%0d quotient=%0d remainder=%0d", $time, a, b, quotient, remainder);
        #10
        a = 11;
        b = 3;
        #10;
        assert (quotient == 3);
        assert (remainder == 2);

        a = 7;
        b = 2;
        #10;
        assert (quotient == 3);
        assert (remainder == 1);

        a = 7;
        b = 3;
        #10;
        assert (quotient == 2);
        assert (remainder == 1);

        a = 123153;
        b = 2424;
        #10;
        assert (quotient == 50);
        assert (remainder == 1953);

        #100 $finish;
    end
endmodule
