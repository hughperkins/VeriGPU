module signed_add();
    reg [31:0] value;
    reg signed [12:0] offset;
    reg [31:0] new_value;

    initial begin
        offset = -3;
        $display("offset %0d", offset);
        value = 100;
        $display("offset %0d value %0d", offset, value);
        new_value = value + offset;
        $display("offset %0d value %0d new_value=%0d", offset, value, new_value);
        new_value = value + {{20{offset[11]}}, offset};
        $display("offset %0d value %0d new_value=%0d", offset, value, new_value);
    end
endmodule
