/*
So, this should contain multiple cores presumably... and somehow manage
communications with gpu controller... but initially we will just run
this directly

For now, we're going to give each core it's own PC, since each core already
has it's own PC, and unclear that that is a terrible idea. If it becomes obvious
that it is a terrible idea, will factorize out the PC from the core at that point
(have a PC layer over the top for running core unit tests; or maybe just use
compute_unit with compute_unit_num_cores set to 1 perhaps?)
*/

parameter compute_unit_num_cores = 8;

module compute_unit(
    // these are all inputs, and can be provided to all cores easily
    // (no need for a multiplexer etc)
    input rst,  // async resets the core, everything goes to zero (do we actually need this?), active low
    input clk,  // clock
    input clr,  // synchronous reset the core; everything goes to zero, active high
    input ena,  // enables gpu to run; active high
    input set_pc_req,  // requests to change PC; best to do this with ena 0
    input [addr_width - 1:0] set_pc_addr, // new address for PC

    // out, outen, outflen, are just to bootstrap development
    // gives a way to write things to console for unit tests
    // we're going to assume that only the first core will write to
    // this, (and maybe only connect the first core)
    output reg [data_width - 1:0] out,
    output reg outen,
    output reg outflen,

    // we will only call this once *all* our cores have halted
    output reg halt,
);
    reg [compute_unit_num_cores - 1:0] core_halt;

    genvar i;
    generate
            core core_(
                .rst(rst),
                .clk(clk),
                .clr(clr),
                .ena(ena),
                .set_pc_req(set_pc_req),
                .set_pc_addr(set_pc_addr),

                .out(out),
                .outen(outen),
                .outflen(outflen),

                .halt(core_halt[0])
            );

        for(i = 1; i < compute_unit_num_cores; i++) begin : core_generate
            core core_(
                .rst(rst),
                .clk(clk),
                .clr(clr),
                .ena(ena),
                .set_pc_req(set_pc_req),
                .set_pc_addr(set_pc_addr),

                .halt(core_halt[i])
            );
        end
    endgenerate
endmodule

/*
    output reg [data_width - 1:0] out,
    output reg outen,
    output reg outflen,

    output reg halt,

    output reg [addr_width - 1:0] mem_addr,
    input [data_width - 1:0]      mem_rd_data,
    output reg [data_width - 1:0] mem_wr_data,
    output reg                    mem_wr_req,
    output reg                    mem_rd_req,
    input                         mem_ack,
    input                         mem_busy
*/