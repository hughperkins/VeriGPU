module always_cycle();
    reg x;
    reg y;

    initial begin
        x = 0;
        #10
        y = 0;
    end

    always @(x, y) begin
        $display("always assign x");
        x = y;
    end

    always @(x, y) begin
        $display("always assign y");
        y = x;
    end
endmodule
