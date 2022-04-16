/*
So, this should contain multiple cores presumably... and somehow manage
communications with gpu controller... but initially we will just run
this directly
*/

parameter compute_unit_num_cores = 8;

module compute_unit(

);
    genvar i;
    generate
        for(i = 0; i < compute_unit_num_cores; i++) begin : core_generate
            core core_(
                
            );
        end
    endgenerate
endmodule
