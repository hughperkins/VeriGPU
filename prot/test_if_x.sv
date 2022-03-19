// examine effect of x on if clauses

module test_if_x();
    reg a;
    reg b;

    always @(*) begin
        assert(~ $isunknown(a));
        if(a == 0) begin
            $display("    a == 0");
        end else begin
            $display("    else for a==0");
        end
        if(a == 1) begin
            $display("    a == 1");
        end else begin
            $display("    else for a==1");
        end
        if(a != 0) begin
            $display("    a != 0");
        end else begin
            $display("    else for a != 0");
        end
        if(a != 1) begin
            $display("    a != 1");
        end else begin
            $display("    else for a != 1");
        end
    end

    initial begin
        $display("a = x");
        a = 1'bx;
        #1;

        $display("a = 0");
        a = 1'b0;
        #1;

        $display("a = 1");
        a = 1'b1;
        #1;

        $display("a = x");
        a = 1'bx;
        #1;
    end
endmodule
