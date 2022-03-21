// functions for help with tests, eg reading/writing floats

function reals_near(real r1, real r2);
    int r1_exp10;
    int r2_exp10;
    int max_exp10;
    real prec;
    real mult;
    int r1_int;
    int r2_int;
    reg sign;

    if(r1 == 0.0 && r2 == 0.0) begin
        // do nothing
        reals_near = 1;
    end else if(r1 < 0 && r2 > 0) begin
        reals_near = 0;
    end else if(r1 > 0 && r2 < 0) begin
        reals_near = 0;
    end else if(r1 == 0 && r2 != 0) begin
        reals_near = 0;
    end else if(r1 != 0 && r2 == 0) begin
        reals_near = 0;
    end else begin
        sign = 0;
        if(r1 < 0) begin
            r1 = - r1;
            r2 = - r2;
            sign = 1;
        end
        r1_exp10 = 0;
        r2_exp10 = 0;
        while(r1 >= 10) begin
            r1 = r1 / 10;
            r1_exp10 += 1;
        end
        while(r2 >= 10) begin
            r2 = r2 / 10;
            r2_exp10 += 1;
        end
        while(r1 < 1) begin
            r1 = r1 * 10;
            r1_exp10 -= 1;
        end
        while(r2 < 1) begin
            r2 = r2 * 10;
            r2_exp10 -= 1;
        end
        max_exp10 = r1_exp10 > r2_exp10 ? r1_exp10 : r2_exp10;

        prec = 0.00001;
        while(max_exp10 > 0) begin
            prec = prec * 10;
            max_exp10 = max_exp10 - 1;
        end
        while(max_exp10 < 0) begin
            prec = prec / 10;
            max_exp10 = max_exp10 + 1;
        end
        mult = 1 / prec;
        r1_int = int'(r1 * mult);
        r2_int = int'(r2 * mult);
        $display("r1 %0f r2 %0f prec %0f mult %0f r1_int %0d r2_int %0d", r1, r2, prec, mult, r1_int, r2_int);
        reals_near = r1_int == r2_int;
    end
endfunction

function [float_width - 1:0] make_float(input real val);
    // given an opaque verilog real, represent it in our own
    // binary representation
    reg sign;
    reg [float_exp_width - 1:0] exp;
    reg [float_mant_width - 1:0] mant;

    if(val == 0.0) begin
        make_float = '0;
    end else begin
        sign = 0;
        $display("%0f", val);
        exp = 127;
        if(val < 0) begin
            val = - val;
            sign = 1;
        end
        while(val >= 2) begin
            val = val / 2;
            exp = exp + 1;
        end
        while(val < 1) begin
            val = val * 2;
            exp = exp - 1;
        end
        val = val - 1;
        for(int i = 0; i < float_mant_width; i = i + 1) begin
            val = val * 2;
        end
        mant = $rtoi(val);
        make_float = {sign, exp, mant};
    end
endfunction

function real to_real(input [float_width - 1:0] fval);
    if(fval == '0) begin
        to_real = '0;
    end else begin
        // given a float in our own representation, convert to opaque verilog real format, and return that
        reg sign;
        reg [float_exp_width - 1:0] exp;
        reg [float_mant_width - 1:0] mant;
        {sign, exp, mant} = fval;

        to_real = $itor(mant);
        // $display("mant as real: %0f", to_real);
        // while(exp > 127) begin
            // to_real = to_real 
        // end
        for(int i = 0; i < 23; i++) begin
            to_real = to_real / 2;
        end
        // $display("mant as real: %0f", to_real);
        to_real = 1 + to_real;
        // $display("mant as real: %0f", to_real);
        while(exp > 127) begin
            exp = exp - 1;
            to_real = to_real * 2;
        end
        while(exp < 127) begin
            exp = exp + 1;
            to_real = to_real / 2;
        end
        // $display("mant as real: %0f", to_real);
        if(sign) begin
            to_real = - to_real;
        end
    end
endfunction
