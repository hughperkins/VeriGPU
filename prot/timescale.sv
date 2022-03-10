`timescale 1ns/10ps
module timescale();
    reg clk;
    initial begin
        clk = 1;
        forever begin
            #0.5 clk=~clk;
        end
    end
    initial begin
        $monitor("clk=%0d t=%0t %0t", clk, $time, $realtime);
        #100 $finish;
    end
endmodule
