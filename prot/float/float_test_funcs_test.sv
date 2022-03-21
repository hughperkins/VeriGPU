// tests for float_test_funcs.sv

module float_test_funcs_test();
    task check_float(input real val_real, input [float_width - 1:0] expected_val_float);
        // converts real val_real, into our representation val_float, then back again
        // check they match. check that the representaion matches passed-in repr
        real reconstr_real;
        reg [float_width - 1:0] val_float;

        val_float = make_float(val_real);
        reconstr_real = to_real(val_float);
        $display("check float: real=%0f reconstr_real=%0f float %b expected float %b", val_real, reconstr_real, val_float, expected_val_float);
        `assert(val_float == expected_val_float);
        // `assert(val_real == reconstr_real);
        `assert(reals_near(val_real, reconstr_real));
    endtask

    initial begin
        // I got the expected values from python, from the test_assembler.py:test_float_to_bits test
        check_float(0.0, 32'b00000000000000000000000000000000);

        check_float(-2.5, 32'b11000000001000000000000000000000);
        check_float(123.456, 32'b01000010111101101110100101111000);

        check_float(4.0, 32'b01000000100000000000000000000000);
        check_float(2.0, 32'b01000000000000000000000000000000);
        check_float(1.0, 32'b00111111100000000000000000000000);
        check_float(0.5, 32'b00111111000000000000000000000000);
        check_float(0.25, 32'b00111110100000000000000000000000);

        check_float(10.123, 32'b01000001001000011111011111001110);
        check_float(123.0, 32'b01000010111101100000000000000000);
        check_float(0.0123, 32'b00111100010010011000010111110000);
        check_float(1.23, 32'b00111111100111010111000010100011);
        check_float(1.2345678, 32'b00111111100111100000011001010001);

        check_float(-10.123, 32'b11000001001000011111011111001110);
        check_float(-123.0, 32'b11000010111101100000000000000000);
        check_float(-0.0123, 32'b10111100010010011000010111110000);
        check_float(-1.23, 32'b10111111100111010111000010100011);
        check_float(-1.2345678, 32'b10111111100111100000011001010001);
    end
endmodule
