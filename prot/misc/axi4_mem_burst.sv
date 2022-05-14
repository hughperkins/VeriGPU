// try playing with axi4
//
// we will start with just adding in the minimum wires we need,
// and then go from there
//
// this version will handle bursts of 4 words

// this will be a memory controller, with an AXI4 interface
// it's thus an axi slave

`default_nettype none

typedef enum bit[1:0] {
    AXI_BRESP_OKAY   = 2'b00,
    AXI_BRESP_EXOKAY = 2'b01,
    AXI_BRESP_SLVERR = 2'b10,
    AXI_BRESP_DECERR = 2'b11
} e_axi_bresp;

typedef enum bit[1:0] {
    AXI_BURST_FIXED = 2'b00,
    AXI_BURST_INCR  = 2'b01,
    AXI_BURST_WRAP  = 2'b10
} e_axi_burst;

module mem #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter MEM_SIZE_BYTES = 4096
)(
    input                         axi_aclk,
    input                         axi_aresetn,

    // read address channel
    input [ADDR_WIDTH - 1:0]      axi_araddr,
    input [2:0]                   axi_arsize,
    input e_axi_burst             axi_arburst,
    input [7:0]                   axi_arlen,
    input                         axi_arvalid,
    output reg                    axi_arready,

    // read data channel
    output reg [DATA_WIDTH - 1:0] axi_rdata,
    output reg                    axi_rlast,
    output reg                    axi_rvalid,
    input                         axi_rready,

    // write address channel
    input [ADDR_WIDTH - 1:0]      axi_awaddr,
    input [2:0]                   axi_awsize,
    input e_axi_burst             axi_awburst,
    input [7:0]                   axi_awlen,
    input                         axi_awvalid,
    output reg                    axi_awready,

    // write data channel
    input [ADDR_WIDTH - 1:0]      axi_wdata,
    input                         axi_wlast,
    input                         axi_wvalid,
    output reg                    axi_wready,

    // write response channel
    output reg                    axi_bresp,
    output reg                    axi_bvalid,
    input                         axi_bready
);
    typedef enum bit[1:0] {
        R_AWAIT_ADDR,
        R_SENDING_DATA,
        R_SENT_DATA
    } e_axi_read_state;

    typedef enum bit[1:0] {
        W_AWAIT_ADDR,
        W_AWAIT_DATA,
        W_SENDING_RESPONSE
    } e_axi_write_state;

    reg e_axi_read_state axi_read_state;
    reg e_axi_write_state axi_write_state;

    reg [ADDR_WIDTH - 1:0] read_addr;
    reg [2:0]              read_size;
    reg e_axi_burst        read_burst;
    reg [7:0]              read_len;
    reg [7:0]              read_remaining;
    // reg                    read_last;

    reg [ADDR_WIDTH - 1:0] write_addr;
    reg [DATA_WIDTH - 1:0] read_data;
    reg [2:0]              write_size;
    reg e_axi_burst        write_burst;
    reg [7:0]              write_len;
    reg [7:0]              write_remaining;

    reg [DATA_WIDTH - 1:0] mem[MEM_SIZE_BYTES >> 2];

    always @(posedge axi_aclk) begin
        // $display("mem posedge clk");
        if(~axi_aresetn) begin
            $display("mem reset");
            axi_arready <= 0;
            axi_rvalid <= 0;
            axi_awready <= 0;
            axi_wready <= 0;
            axi_bvalid <= 0;
            axi_read_state <= R_AWAIT_ADDR;
            axi_write_state <= W_AWAIT_ADDR;
        end else begin
            axi_rvalid <= 0;
            axi_arready <= 0;
            // axi_rlast <= 0;
            case(axi_read_state)
                R_AWAIT_ADDR: begin
                    axi_arready <= 1;
                    if(axi_arready & axi_arvalid) begin
                        $display("got read address %0d", axi_araddr);
                        read_addr <= axi_araddr;

                        read_burst <= axi_arburst;
                        read_size <= (1 << axi_arsize);
                        read_len <= axi_arlen;
                        read_remaining <= axi_arlen;
                        $display("axi_arburst %0d", axi_arburst);
                        assert(axi_arburst == AXI_BURST_INCR);
                        $display("axi_arsize %0d", axi_arsize);
                        $display("axi_arlen %0d", axi_arlen);
                        assert((1 << axi_arsize) == 4);  // word at a time
                        assert(axi_arlen == 4);      // 4 words

                        axi_read_state <= R_SENDING_DATA;
                        $display("axi_arready <= 0");
                        axi_arready <= 0;
                        axi_rvalid <= 1;
                        read_data <= mem[axi_araddr];
                        axi_rdata <= mem[axi_araddr];
                        read_addr <= axi_araddr + (1 << axi_arsize);
                        if(axi_arlen == 1) begin
                            axi_rlast <= 1;
                        end else begin
                            axi_rlast <= 0;
                        end
                    end
                end
                R_SENDING_DATA: begin
                    axi_rdata <= read_data;
                    axi_rvalid <= 1;
                    if(axi_rready & axi_rvalid) begin
                        $display(
                            "valid ready read mem read_addr=%0d read_remaining=%0d",
                            read_addr, read_remaining);
                        if(read_remaining != 1) begin
                            read_data <= mem[read_addr];
                            axi_rdata <= mem[read_addr];
                            read_addr <= read_addr + (1 << axi_arsize);
                            read_remaining <= read_remaining - 1;
                            if(read_remaining == 2) begin
                                axi_rlast <= 1;
                            end
                        end
                        if(read_remaining == 1) begin
                            // axi_rlast <= 1;
                            $display("mem finished burst read");
                            axi_rvalid <= 0;
                            axi_read_state <= R_AWAIT_ADDR;
                            axi_arready <= 1;
                        end
                    end
                    // end
                end
            endcase

            axi_awready <= 0;
            axi_wready <= 0;
            case(axi_write_state)
                W_AWAIT_ADDR: begin
                    axi_awready <= 1;
                    if(axi_awready & axi_awvalid) begin
                        $display("got write address %0d", axi_awaddr);
                        write_addr <= axi_awaddr;
                        write_burst <= axi_awburst;
                        write_size <= (1 << axi_awsize);
                        write_len <= axi_awlen;
                        write_remaining <= axi_awlen;
                        $display("axi_awburst %0d", axi_awburst);
                        assert(axi_awburst == AXI_BURST_INCR);
                        $display("axi_awsize %0d", axi_awsize);
                        $display("axi_awlen %0d", axi_awlen);
                        assert((1 << axi_awsize) == 4);  // word at a time
                        assert(axi_awlen == 4);      // 4 words

                        axi_write_state <= W_AWAIT_DATA;
                        axi_awready <= 0;
                        axi_wready <= 1;
                    end
                end
                W_AWAIT_DATA: begin
                    axi_wready <= 1;
                    if(axi_wready & axi_wvalid) begin
                        $display("write %0d to mem[%0d]", axi_wdata, write_addr);
                        mem[write_addr] <= axi_wdata;
                        write_remaining <= write_remaining - 1;
                        write_addr <= write_addr + write_size;

                        if(write_remaining == 1) begin
                            $display("got final write");
                            assert(axi_wlast);
                            axi_write_state <= W_SENDING_RESPONSE;
                            axi_wready <= 0;
                            axi_bresp <= AXI_BRESP_OKAY;
                            axi_bvalid <= 1;
                        end
                    end
                end
                W_SENDING_RESPONSE: begin
                    axi_bresp <= AXI_BRESP_OKAY;
                    axi_bvalid <= 1;
                    if(axi_bvalid & axi_bready) begin
                        axi_bvalid <= 0;
                        axi_write_state <= W_AWAIT_ADDR;
                        axi_awready <= 1;
                    end
                end
            endcase
        end
    end
endmodule

// this will read and write memroy, via axi4. It will be
// an axi4 master
module dut #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    input                         axi_aclk,
    input                         axi_aresetn,

    // read address channel
    output reg [ADDR_WIDTH - 1:0] axi_araddr,
    output reg [2:0]              axi_arsize,
    output reg [1:0]              axi_arburst,
    output reg [7:0]              axi_arlen,
    output reg                    axi_arvalid,
    input                         axi_arready,

    // read data channel
    input [DATA_WIDTH - 1:0]      axi_rdata,
    input                         axi_rlast,
    input                         axi_rvalid,
    output reg                    axi_rready,

    // write address channel
    output reg [ADDR_WIDTH - 1:0] axi_awaddr,
    output reg                    axi_awvalid,
    input                         axi_awready,

    // write data channel
    output reg [ADDR_WIDTH - 1:0] axi_wdata,
    output reg [2:0]              axi_awsize,
    output reg [1:0]              axi_awburst,
    output reg [7:0]              axi_awlen,
    output reg                    axi_wlast,
    output reg                    axi_wvalid,
    input                         axi_wready,

    // write response channel
    input                         axi_bresp,
    input                         axi_bvalid,
    output reg                    axi_bready
);
    reg [7:0] write_remaining;
    reg [7:0] read_remaining;

    task initiate_burst_write([ADDR_WIDTH - 1:0] addr, [7:0] len);
        axi_awaddr <= addr;
        axi_awburst <= AXI_BURST_INCR;
        axi_awsize <= $clog2(4);  // one word at a time
        axi_awlen <= len;
        axi_awvalid <= 1;
        write_remaining <= len;
        assert(~axi_awvalid);
        assert(~axi_awready | ~axi_awvalid);
        assert(~axi_wready | ~axi_wvalid);

        #10
        assert(axi_awready & axi_awvalid);
        assert(~axi_wready | ~axi_wvalid);
        axi_awvalid <= 0;
    endtask

    task write_mem([DATA_WIDTH - 1:0] data);
        $display("write_mem %0d", data);
        axi_wlast <= 0;
        write_remaining <= write_remaining - 1;
        axi_wdata <= data;
        axi_wvalid <= 1;
        if(write_remaining == 1) begin
            $display("master wrote all");
            axi_wlast <= 1;
        end

        #10;
        assert(axi_wready & axi_wvalid);
        if(write_remaining == 0) begin
            axi_wvalid <= 0;

            axi_bready <= 1;
            
            #10
            assert(axi_bvalid & axi_bready);
            assert(~axi_wready & ~axi_wvalid);
            axi_bready <= 0;
        end
    endtask

    task initiate_burst_read([ADDR_WIDTH - 1:0] addr, [7:0] len);
        $display("initiate burst read");
        axi_araddr <= addr;
        axi_arburst <= AXI_BURST_INCR;
        axi_arsize <= $clog2(4);  // one word at a time
        axi_arlen <= len;
        axi_arvalid <= 1;
        read_remaining <= len;
        assert(~axi_arvalid);

        #10;
        assert(axi_arready);
        assert(axi_arvalid);

        axi_arvalid <= 0;
    
        axi_rready <= 1;
    endtask

    task read_mem([DATA_WIDTH - 1:0] expected_data);
        #10;
        assert(~axi_arready);
        assert(~axi_arvalid);

        assert(axi_rvalid & axi_rready);
        $display("read_mem expected=%0d read=%0d from mem", expected_data, axi_rdata);
        assert(axi_rdata == expected_data);
        read_remaining <= read_remaining - 1;
        if(read_remaining == 1) begin
            $display("master read last");
            assert(axi_rlast);
            axi_rready <= 0;
        end

        if(read_remaining == 1) begin
            $display("master finished read");
        end
    endtask

    initial begin
        // $monitor(
        //     "axi_awready=%0d axi_awvalid=%0d axi_wready=%0d axi_wvalid=%0d",
        //     axi_awready, axi_awvalid, axi_wready, axi_wvalid);
        // $monitor(
        //     "axi_arready=%0d axi_arvalid=%0d axi_rready=%0d axi_rvalid=%0d",
        //     axi_arready, axi_arvalid, axi_rready, axi_rvalid);
        #40;
        $display("axi_aresetn %d", axi_aresetn);

        initiate_burst_write(20, 4);
        write_mem(123);
        write_mem(234);
        write_mem(222);
        write_mem(333);

        initiate_burst_write(48, 4);
        write_mem(555);
        write_mem(444);
        write_mem(333);
        write_mem(111);

        initiate_burst_read(20, 4);
        read_mem(123);
        read_mem(234);
        read_mem(222);
        read_mem(333);

        initiate_burst_read(48, 4);
        read_mem(555);
        read_mem(444);
        read_mem(333);
        read_mem(111);
    end

    always @(posedge axi_aclk) begin
        
    end

    always @(posedge axi_aclk) begin
        if(~axi_aresetn) begin
            axi_arvalid <= 0;
            axi_rready <= 0;

            axi_awvalid <= 0;
            axi_wvalid <= 0;
        end
    end
endmodule

module test_dut #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)();
    reg axi_aclk;
    reg axi_aresetn;

    // read address channel
    reg [ADDR_WIDTH - 1:0] axi_araddr;
    reg [2:0]                   axi_arsize;
    reg e_axi_burst             axi_arburst;
    reg [7:0]                   axi_arlen;
    reg axi_arvalid;
    reg axi_arready;

    // read data channel
    reg [DATA_WIDTH - 1:0] axi_rdata;
    reg axi_rlast;
    reg axi_rvalid;
    reg axi_rready;

    // write address channel
    reg [ADDR_WIDTH - 1:0] axi_awaddr;
    reg [2:0]                   axi_awsize;
    reg e_axi_burst             axi_awburst;
    reg [7:0]                   axi_awlen;
    reg axi_awvalid;
    reg axi_awready;

    // write data channel
    reg [DATA_WIDTH - 1:0] axi_wdata;
    reg axi_wlast;
    reg axi_wvalid;
    reg axi_wready;

    // write response channel
    reg axi_bresp;
    reg axi_bvalid;
    reg axi_bready;

    dut dut_(
        .axi_aclk(axi_aclk),
        .axi_aresetn(axi_aresetn),

        .axi_araddr(axi_araddr),
        .axi_arburst(axi_arburst),
        .axi_arsize(axi_arsize),
        .axi_arlen(axi_arlen),
        .axi_arvalid(axi_arvalid),
        .axi_arready(axi_arready),

        .axi_rdata(axi_rdata),
        .axi_rlast(axi_rlast),
        .axi_rvalid(axi_rvalid),
        .axi_rready(axi_rready),

        .axi_awaddr(axi_awaddr),
        .axi_awburst(axi_awburst),
        .axi_awsize(axi_awsize),
        .axi_awlen(axi_awlen),
        .axi_awvalid(axi_awvalid),
        .axi_awready(axi_awready),

        .axi_wdata(axi_wdata),
        .axi_wlast(axi_wlast),
        .axi_wvalid(axi_wvalid),
        .axi_wready(axi_wready),

        .axi_bresp(axi_bresp),
        .axi_bvalid(axi_bvalid),
        .axi_bready(axi_bready)
    );
    mem mem_(
        .axi_aclk(axi_aclk),
        .axi_aresetn(axi_aresetn),

        .axi_araddr(axi_araddr),
        .axi_arburst(axi_arburst),
        .axi_arsize(axi_arsize),
        .axi_arlen(axi_arlen),
        .axi_arvalid(axi_arvalid),
        .axi_arready(axi_arready),

        .axi_rdata(axi_rdata),
        .axi_rlast(axi_rlast),
        .axi_rvalid(axi_rvalid),
        .axi_rready(axi_rready),

        .axi_awaddr(axi_awaddr),
        .axi_awburst(axi_awburst),
        .axi_awsize(axi_awsize),
        .axi_awlen(axi_awlen),
        .axi_awvalid(axi_awvalid),
        .axi_awready(axi_awready),

        .axi_wdata(axi_wdata),
        .axi_wlast(axi_wlast),
        .axi_wvalid(axi_wvalid),
        .axi_wready(axi_wready),

        .axi_bresp(axi_bresp),
        .axi_bvalid(axi_bvalid),
        .axi_bready(axi_bready)
    );

    initial begin
        $display("clk initial");
        axi_aclk = 1;
        forever begin
            #5 axi_aclk = ~axi_aclk;
            if(axi_aclk) begin
                $display("");
                $display("positive clock edge");
            end
        end
    end

    initial begin
        $dumpfile("test.vcd");
        $dumpvars(0,test_dut);
    end
    
    initial begin
        $display("reset initial");
        axi_aresetn <= 0;
        #20
        axi_aresetn <= 1;
        #280 $finish;
    end
endmodule
