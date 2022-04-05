; this file was generated from prot/verilator/prot_unified_source/my_gpu_test_client.cpp by
; first using clang++     -x cuda --cuda-device-only -emit-llvm -nocudainc -nocudalib
; to get a .ll file
; then using llc  --march=riscv32 to get this assembly :)
; then some lines were added just below this header text to wrap this, and make it directly
; runnable

; here was the original kernel code that was converted like this into riscv32 assmebler belwo:
; __global__ void sum_ints(unsigned  int *in, unsigned int numInts, unsigned int *p_out) {
;     // sum the ints in in, and write the result to *out
;     // we assume just a single thread/core for now
;     unsigned int out = 0;
;     for(unsigned int i = 0; i < numInts; i++) {
;         out += in[i];
;     }
;     *p_out = out;
; }

; so we have three args:
; - pointer to ints
; - number of ints
; - pointer to output sum

; lets have three ints

li sp, 1000
addi sp, sp, -64
li a0, 5
sw a0, 60(sp)
li a0, 12
sw a0, 56(sp)
li a0, 9
sw a0, 52(sp)
addi a0, sp, 52
li a1, 3
addi a2, sp, 48

; so, we created a stack
; moved stack pointer down a bit
; put some ints onto the staack
; pointed a0 at these ints
; set a1 to 3
; pointed a2 at anotherspace on this stack

; call _Z8sum_intsPjjS_
; note to self: need to fix call
; answer should be 5 + 12 + 9 = 17 + 9 = 26
jal x1, _Z8sum_intsPjjS_

lw a0, 48(sp)
outr a0
halt

	.text
	.attribute	4, 16
	.attribute	5, "rv32i2p0"
	.file	"my_gpu_test_client.cpp"
	.globl	_Z8sum_intsPjjS_                # -- Begin function _Z8sum_intsPjjS_
	.p2align	2
	.type	_Z8sum_intsPjjS_,@function
_Z8sum_intsPjjS_:                       # @_Z8sum_intsPjjS_
# %bb.0:
	addi	sp, sp, -32
	sw	ra, 28(sp)                      # 4-byte Folded Spill
	sw	s0, 24(sp)                      # 4-byte Folded Spill
	addi	s0, sp, 32
	sw	a0, -16(s0)
	sw	a1, -20(s0)
	sw	a2, -24(s0)
	sw	zero, -28(s0)
	sw	zero, -32(s0)
	j	.LBB0_1
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	lw	a0, -32(s0)
	lw	a1, -20(s0)
	bgeu	a0, a1, .LBB0_4
	j	.LBB0_2
.LBB0_2:                                #   in Loop: Header=BB0_1 Depth=1
	lw	a0, -16(s0)
	lw	a1, -32(s0)
	slli	a1, a1, 2
	add	a0, a0, a1
	lw	a0, 0(a0)
	lw	a1, -28(s0)
	add	a0, a1, a0
	sw	a0, -28(s0)
	j	.LBB0_3
.LBB0_3:                                #   in Loop: Header=BB0_1 Depth=1
	lw	a0, -32(s0)
	addi	a0, a0, 1
	sw	a0, -32(s0)
	j	.LBB0_1
.LBB0_4:
	lw	a0, -28(s0)
	lw	a1, -24(s0)
	sw	a0, 0(a1)
	lw	ra, 28(sp)                      # 4-byte Folded Reload
	lw	s0, 24(sp)                      # 4-byte Folded Reload
	addi	sp, sp, 32
	ret
.Lfunc_end0:
	.size	_Z8sum_intsPjjS_, .Lfunc_end0-_Z8sum_intsPjjS_
                                        # -- End function
	.ident	"clang version 14.0.0 (https://github.com/tru/llvm-release-build fc075d7c96fe7c992dde351695a5d25fe084794a)"
	.section	".note.GNU-stack","",@progbits