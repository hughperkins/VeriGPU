/*
This will manage communicaitons with the cpu/driver.

Will handle things like:
- allocating gpu memory
- copy data to/from gpu
- copy kernel to gpu
- launching kernels

for now, we assume `instr` is directly readable somehow.

Since there won't be many instructions, and since splitting data and addresses
across two instructions is kind of a PITA, we're going to use entire 32-bit words
for each data or address provided with an instruction. If we include such things in
the concept of 'instruction', then the instructions are therefore variable-length.

things we need to handle:
- mmemory alloc. actually, no. we'll let the driver handle this
- memory free. ditto. no instructions to controller for this
- copy data to gpu memory. we'll need
    - destination address (in bytes)
    - number of data (in bytes)
    - then in subsequent clock cycles will receive the data
- copy out data from gpu memory. we'll need
    - gpu address (in bytes)
    - number of data (in bytes)
    - then in subsequent clock cycles will send the data
- launch kernel. we'll need
    - gpu address of kernel (in bytes)
    - maybe a time-out, in thousands of cycles, for running the kernel???
- maybe abort kernel???
- maybe reset???

- let's just always have three words of parameters following each instruction word?

We can be pretty wasteful of resources in the controller, since there's only one of them on the die.
(cf thousands of cores...)
*/
module gpu_controller(
    input clk,
    input rst,

    // comms with mainboard cpu
    input [31:0] cpu_recv_instr,  
    // I'm using in/out, because less ambigous than rd/wr I feel, i.e. invariant
    // with PoV this module, or PoV calling module
    input [31:0] cpu_in_data,
    output reg [31:0] cpu_out_data,
    output reg cpu_out_ack,  // high each time we are outputing data

    // we also need to be able to read/write memory
    output reg mem_wr_en,
    output reg mem_rd_en,
    output reg [addr_width - 1:0] mem_wr_addr,
    output reg [data_width - 1:0] mem_wr_data,
    output reg [addr_width - 1:0] mem_rd_addr,
    input [data_width - 1:0] mem_rd_data,
    input mem_rd_ack,

    // and core (later: cores)
    output reg core_en,
    output reg core_clr,
    output reg core_set_pc_req,
    output reg [data_width - 1:0] core_set_pc_addr
);
    parameter MAX_PARAMS = 4;

    reg [5:0] state;
    reg [31:0] instr;
    reg [$clog2(MAX_PARAMS) - 1:0] param_pos;
    reg [$clog2(MAX_PARAMS) - 1:0] num_params;
    reg [31:0] params [MAX_PARAMS];
    // used to track which data to read/write next
    reg [31:0] internal_data_addr;
    // we read or write data until data_addr equals last_data_addr_excl
    // this means we dont have to decrement a counter, as well as incrementing
    // data_addr
    // used to store one past the end address to read/write
    reg [31:0] internal_end_data_addr_excl;
    // reg [31:0] data_cnt;

    // reg mem_wr_req;
    // reg mem_rd_req;
    // reg [31:0] mem_wr_addr;
    // reg [31:0] mem_rd_addr;
    // reg [31:0] mem_wr_data;
    // reg [31:0] mem_rd_data;

    reg n_mem_wr_en;
    reg n_mem_rd_en;
    reg [addr_width - 1:0] n_mem_wr_addr;
    reg [data_width - 1:0] n_mem_wr_data;
    reg [addr_width - 1:0] n_mem_rd_addr;

    reg [5:0] n_state;
    reg [31:0] n_instr;
    reg [$clog2(MAX_PARAMS) - 1:0] n_param_pos;
    reg [$clog2(MAX_PARAMS) - 1:0] n_num_params;
    reg [31:0] n_params [MAX_PARAMS];

    reg [31:0] n_internal_data_addr;
    reg [31:0] n_internal_end_data_addr_excl;
    // reg [31:0] n_data_cnt;

    reg internal_mem_read_sent;
    reg n_internal_mem_read_sent;
    // reg [31:0] internal_last_mem_read_addr;
    // reg [31:0] n_internal_last_mem_read_addr;
    // reg mem_read

    // reg n_mem_wr_req;
    // reg n_mem_rd_req;
    // reg [31:0] n_mem_wr_addr;
    // reg [31:0] n_mem_rd_addr;
    // reg [31:0] n_mem_wr_data;
    // reg [31:0] n_mem_rd_data;

    reg [31:0] n_out_data;
    reg n_cpu_out_ack;

    typedef enum bit[5:0] {
        IDLE,
        RECV_PARAMS,
        RECEIVE_DATA,
        SEND_DATA
    } e_state;

    typedef enum bit[31:0] {
        NOP = 0,
        COPY_TO_GPU = 1,
        COPY_FROM_GPU = 2,
        KERNEL_LAUNCH = 3
    } e_instr;

    // reg [31:0] mem [512];  // put here for now, use comp's in a bit

    always @(*) begin
        n_state = state;
        n_instr = instr;
        n_param_pos = '0;
        n_out_data = '0;
        n_internal_end_data_addr_excl = internal_end_data_addr_excl;
        n_internal_data_addr = internal_data_addr;

        // n_mem_rd_req = 0;
        // n_mem_wr_req = 0;
        // n_mem_rd_addr = '0;
        // n_mem_wr_addr = '0;
        // n_mem_wr_data = '0;

        n_internal_mem_read_sent = internal_mem_read_sent;

        n_mem_wr_en = 0;
        n_mem_rd_en = 0;
        n_mem_wr_addr = '0;
        n_mem_wr_data = '0;
        n_mem_rd_addr = '0;
        n_cpu_out_ack = 0;

        for(int i = 0; i < MAX_PARAMS; i++) begin
            n_params[i] = '0; 
        end

        if(rst) begin
            case(state)
                IDLE: begin
                    n_state = RECV_PARAMS;
                    n_param_pos = 0;
                    n_instr = cpu_recv_instr;
                    case(cpu_recv_instr)
                        COPY_TO_GPU: begin
                            $display("gpucontroller COPY_TO_GPU");
                            n_num_params = 2;
                        end
                        COPY_FROM_GPU: begin
                            $display("gpucontroller COPY_FROM_GPU");
                            n_num_params = 2;
                        end
                        NOP: begin
                            $display("NOP");
                            n_state = IDLE;
                            // do nothing...
                        end
                        default: begin
                            $display("gpucontroller case recv_instr hit default");
                        end
                    endcase
                end
                RECV_PARAMS: begin
                    $display("gpucontroller RECV_PARAMS param_pos=%0d in_data=%0d", param_pos, cpu_in_data);
                    // we use in_data to receive because
                    // means we can give more control to recv_instr, eg it could
                    // send RESET in the middle of sending a new instruction, and we wouldn't
                    // have issues with going 'out of sync': interpreting params as instruction
                    // (cf sending parmetres via recv_instr wires)
                    n_params[param_pos] = cpu_in_data;
                    n_param_pos = param_pos + 1;
                    if(n_param_pos == num_params) begin
                        case(instr)
                            COPY_TO_GPU: begin
                                n_internal_data_addr = params[0];
                                n_internal_end_data_addr_excl = params[0] + n_params[1];
                                $display("gpucontroller RECV_PARAMS COPY_TO_GPU addr %0d count %0d final_addr_excl %0d", params[0], n_params[1], n_internal_end_data_addr_excl);
                                n_state = RECEIVE_DATA;
                            end
                            COPY_FROM_GPU: begin
                                n_internal_end_data_addr_excl = params[0] + n_params[1];
                                $display("gpucontroller RECV_PARAMS COPY_FROM_GPU addr %0d count %0d final_addr_excl %0d", params[0], n_params[1], n_internal_end_data_addr_excl);
                                n_state = SEND_DATA;
                                // n_mem_rd_addr = params[0];
                                n_mem_rd_en = 1;
                                n_mem_rd_addr = params[0];
                                // n_internal_last_mem_read_addr = params[0];
                                n_internal_mem_read_sent = 1;
                                n_internal_data_addr = params[0] + 4;
                                // n_mem_rd_req =? 1;
                            end
                            default: begin
                                $display("recv params case instr hit default");
                            end
                        endcase
                    end
                end
                RECEIVE_DATA: begin
                    $display("gpucontroller RECEIVE_DATA");
                    $display("gpucontroller receive data addr %0d val %0d", internal_data_addr, cpu_in_data);
                    // mem[data_addr] = in_data;
                    // n_mem_wr_en = 1;
                    // n_mem_wr_addr = data_addr;
                    // n_mem_wr_data = cpu_in_data;

                    n_mem_wr_en = 1;
                    n_mem_wr_addr = internal_data_addr;
                    n_mem_wr_data = cpu_in_data;

                    n_internal_data_addr = internal_data_addr + 4;

                    if(n_internal_data_addr >= internal_end_data_addr_excl) begin
                        $display("gpucontroller finished data receive");
                        n_state = IDLE;
                    end
                end
                SEND_DATA: begin
                    $display("gpucontroller SEND_DATA");
                    if(internal_mem_read_sent && mem_rd_ack) begin
                        n_out_data = mem_rd_data;
                        n_internal_mem_read_sent = 0;
                        $display("gpucontroller send data %0d", mem_rd_data);
                        n_cpu_out_ack = 1;
                        if(n_internal_data_addr < internal_end_data_addr_excl) begin
                            n_mem_rd_en = 1;
                            n_mem_rd_addr = internal_data_addr;
                            n_internal_mem_read_sent = 1;
                            n_internal_data_addr = internal_data_addr + 4;
                            // n_internal_last_mem_read_addr = data_addr;
                            // n_out_data = mem[data_addr];
                            // n_out_data = mem_rd_data;
                        end
                    end

                    if(mem_rd_ack && ~n_internal_mem_read_sent) begin
                        $display("gpucontroller finished data send => IDLE");
                        n_state = IDLE;
                    end
                    // if(n_data_addr >= last_data_addr_excl) begin
                    // end else begin
                    //     n_mem_rd_req = 1;
                    //     n_mem_rd_addr = n_data_addr;
                    // end
                end
                default: begin
                    $display("gpucontroller case state hit default");
                end
            endcase
        end
    end
    always @(posedge clk, negedge rst) begin
        // $display("controller.ff");
        if(~rst) begin
            cpu_out_data <= '0;
            state <= IDLE;
            param_pos <= '0;
            instr <= '0;
            num_params <= '0;

            for(int i = 0; i < MAX_PARAMS; i++) begin
               params[i] <= '0; 
            end

            internal_data_addr <= '0;
            internal_end_data_addr_excl <= '0;

            mem_wr_en <= 0;
            mem_rd_en <= 0;
            mem_wr_addr <= '0;
            mem_wr_data <= '0;
            mem_rd_addr <= '0;

            internal_mem_read_sent <= 0;

            core_en <= 0;
            core_clr <= 0;
            core_set_pc_req <= 0;
            core_set_pc_addr <= '0;

            cpu_out_ack = 0;

            // mem_rd_data <= '0;
        end else begin
            cpu_out_data <= n_out_data;
            state <= n_state;
            instr <= n_instr;
            param_pos <= n_param_pos;
            num_params <= n_num_params;

            for(int i = 0; i < MAX_PARAMS; i++) begin
               params[i] <= n_params[i]; 
            end

            internal_data_addr <= n_internal_data_addr;
            internal_end_data_addr_excl <= n_internal_end_data_addr_excl;

            mem_wr_en <= n_mem_wr_en;
            mem_rd_en <= n_mem_rd_en;
            mem_wr_addr <= n_mem_wr_addr;
            mem_wr_data <= n_mem_wr_data;
            mem_rd_addr <= n_mem_rd_addr;

            internal_mem_read_sent <= n_internal_mem_read_sent;

            cpu_out_ack = n_cpu_out_ack;

            // if(n_mem_wr_req) begin
            //     $display("ff write mem addr=%0d data=%0d", n_mem_wr_addr, n_mem_wr_data);
            //     mem[n_mem_wr_addr] <= n_mem_wr_data;
            // end
            // if(n_mem_rd_req) begin
            //     $display("ff read mem addr=%0d data=%0d", n_mem_rd_addr, mem[n_mem_rd_addr]);
            //     mem_rd_data <= mem[n_mem_rd_addr];
            // end
        end
    end
endmodule
