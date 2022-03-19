
`define assert_known(VAL) \
    if(~rst & $isunknown(VAL)) begin \
        $display("unknown value at %s line %0d", `__FILE__, `__LINE__); \
        $fatal(); \
    end

    //  $display("assert known rst=%0d val=%0d", rst, VAL); \

// fatally dies immediately, instead of carrying on running
`define assert(VAL) \
    if(~(VAL)) begin \
        $display("failed assert at %s line %0d", `__FILE__, `__LINE__); \
        $fatal(); \
    end
