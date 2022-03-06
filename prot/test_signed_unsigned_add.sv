module test_signed_unsigned_add();
    reg [31:0] value;
    reg signed [31:0] offset;
    reg signed [31:0] half_offset;
    reg [31:0] new_value;

    initial begin
        value = 4;
        offset = -6;
        half_offset = offset >>> 1;
        new_value = value + (offset >>> 1);
        new_value = value + {offset[31], offset[31:1]};
        // new_value = value + half_offset;
        $display("offset %0d value %0d new_value=%0d half_offset=%0d", offset, value, new_value, half_offset);
    end
endmodule
