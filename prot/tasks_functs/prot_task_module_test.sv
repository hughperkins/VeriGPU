// test calling prot_task_module
module prot_task_module_test();
    reg [1:0] a;
    reg [1:0] b;
    reg [1:0] out;

    prot_task_module prot_task_(
        .a(a),
        .b(b),
        .out(out)
    );

    initial begin
        a = 0;
        b = 1;
        #1;
        `assert(out == 1);

        a = 2;
        b = 1;
        #1;
        `assert(out == 3);

        a = 3;
        b = 1;
        #1;
        `assert(out == 0);

        a = 3;
        b = 2;
        #1;
        `assert(out == 1);

        $display("done");
    end
endmodule
