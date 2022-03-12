module test_int_div_pipeline();
    reg [bitwidth - 1:0] a;
    reg [bitwidth - 1:0] b;
    reg [bitwidth - 1:0] quotient;
    reg [bitwidth - 1:0] remainder;
    reg clk;
    reg req;
    reg ack;

    int_div_pipeline dut(.ack(ack), .req(req), .clk(clk), .a(a), .b(b), .quotient(quotient), .remainder(remainder) );

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        $monitor("t=%0d a=%0d b=%0d quotient=%0d remainder=%0d ack=%b", $time, a, b, quotient, remainder, ack);
        req = 1;
        a = 11;
        b = 3;

        #10;
        req = 0;

        #350
        assert (quotient == 3);
        assert (remainder == 2);

        req = 1;
        a = 7;
        b = 2;

        #10
        req = 0;
        #350
        assert (quotient == 3);
        assert (remainder == 1);

        req = 1;
        a = 7;
        b = 3;

        #10
        req = 0;
        #350
        assert (quotient == 2);
        assert (remainder == 1);

        req = 1;
        a = 123153;
        b = 2424;

        #10
        req = 0;
        #350
        assert (quotient == 50);
        assert (remainder == 1953);

        #500 $finish;
    end
endmodule
