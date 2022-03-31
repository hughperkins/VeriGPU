; generated from prot/cpp/prime_numbers_cpp.cpp by doing
; clang prot/cpp/prime_numbers_cpp.cpp -S -o test.ll -target riscv32
; then adding these headers lines, to test.ll, down to line 11 below, by hand
li sp, 800
addi sp, sp, -16
jal x1, _Z17prime_numbers_cppv
halt

_Z3outj:
    outr a0
    ret


	.text
	.attribute	4, 16
	.attribute	5, "rv32i2p0_m2p0_a2p0_c2p0"
	.file	"prime_numbers_cpp.cpp"
	.globl	_Z17prime_numbers_cppv
	.p2align	1
	.type	_Z17prime_numbers_cppv,@function
_Z17prime_numbers_cppv:
	.cfi_startproc
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	sw	ra, 28(sp)
	sw	s0, 24(sp)
	.cfi_offset ra, -4
	.cfi_offset s0, -8
	addi	s0, sp, 32
	.cfi_def_cfa s0, 0
	li	a0, 2
	sw	a0, -12(s0)
	j	.LBB0_1
.LBB0_1:
	lw	a1, -12(s0)
	li	a0, 31
	bltu	a0, a1, .LBB0_12
	j	.LBB0_2
.LBB0_2:
	li	a0, 1
	sw	a0, -16(s0)
	li	a0, 2
	sw	a0, -20(s0)
	j	.LBB0_3
.LBB0_3:
	lw	a0, -20(s0)
	lw	a1, -12(s0)
	bgeu	a0, a1, .LBB0_8
	j	.LBB0_4
.LBB0_4:
	lw	a0, -12(s0)
	lw	a1, -20(s0)
	remu	a0, a0, a1
	li	a1, 0
	bne	a0, a1, .LBB0_6
	j	.LBB0_5
.LBB0_5:
	li	a0, 0
	sw	a0, -16(s0)
	j	.LBB0_8
.LBB0_6:
	j	.LBB0_7
.LBB0_7:
	lw	a0, -20(s0)
	addi	a0, a0, 1
	sw	a0, -20(s0)
	j	.LBB0_3
.LBB0_8:
	lw	a0, -16(s0)
	li	a1, 0
	beq	a0, a1, .LBB0_10
	j	.LBB0_9
.LBB0_9:
	lw	a0, -12(s0)
	call	_Z3outj
	j	.LBB0_10
.LBB0_10:
	j	.LBB0_11
.LBB0_11:
	lw	a0, -12(s0)
	addi	a0, a0, 1
	sw	a0, -12(s0)
	j	.LBB0_1
.LBB0_12:
	lw	ra, 28(sp)
	lw	s0, 24(sp)
	addi	sp, sp, 32
	ret
.Lfunc_end0:
	.size	_Z17prime_numbers_cppv, .Lfunc_end0-_Z17prime_numbers_cppv
	.cfi_endproc

	.ident	"clang version 14.0.0 (https://github.com/tru/llvm-release-build fc075d7c96fe7c992dde351695a5d25fe084794a)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z3outj
