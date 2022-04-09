; this was created by using clang against a .cpp file, and then prefixing this file with some asembly to:
; - run the intended function (eg _Z3fooii)
; - have an _Z3outi function
; 
; These are first steps to being able to directly compile and execute a c++ file
; 
; ~/Downloads/clang+llvm-14.0.0-x86_64-apple-darwin/bin/clang++ prot/cpp/test.cpp -S -o test.ll -target riscv32
;
; things we need in the assembler:
; - aliases for registers:
;    sp == x2
;    ra == x1
;    a0-7 = x10-x17
; - ret instruction
; - call pseudo instruction
;   call offset
;   auipc x1, offset[31 : 12] + offset[11] Call far-away subroutine
;   jalr x1, offset[11:0](x1)
;   (could initially make it an alias for jalr)
; - some kind of jump instruction (JAL?)
; - ignore things starting wtih .
; some way of deciding which function name to jump to

li sp, 1000

addi sp, sp, -16

li a0, 123
li a1, 2222

jal x1, _Z3fooii
out 999

li a0, 99
sw a0, 4(sp)

li a0, 3
li a1, 7
addi a2, sp, 4

jal x1, _Z4foo2iiPi
lw a0, 4(sp)
outr a0

finish:
halt

_Z3outi:
    outr a0
    ret

    .text
    .attribute  4, 16
    .attribute  5, "rv32i2p0_m2p0_a2p0_c2p0"
    .file   "test.cpp"
    .globl  _Z3fooii
    .p2align    1
    .type   _Z3fooii,@function
_Z3fooii:
    .cfi_startproc
    addi    sp, sp, -16
    .cfi_def_cfa_offset 16
    sw  ra, 12(sp)
    sw  s0, 8(sp)
    .cfi_offset ra, -4
    .cfi_offset s0, -8
    addi    s0, sp, 16
    .cfi_def_cfa s0, 0
    sw  a0, -12(s0)
    sw  a1, -16(s0)
    lw  a0, -12(s0)
    call    _Z3outi
    lw  a0, -16(s0)
    call    _Z3outi
    lw  a0, -12(s0)
    lw  a1, -16(s0)
    mul a0, a0, a1
    lw  ra, 12(sp)
    lw  s0, 8(sp)
    addi    sp, sp, 16
    ret
.Lfunc_end0:
    .size   _Z3fooii, .Lfunc_end0-_Z3fooii
    .cfi_endproc

    .globl  _Z4foo2iiPi
    .p2align    1
    .type   _Z4foo2iiPi,@function
_Z4foo2iiPi:
    .cfi_startproc
    addi    sp, sp, -32
    .cfi_def_cfa_offset 32
    sw  ra, 28(sp)
    sw  s0, 24(sp)
    .cfi_offset ra, -4
    .cfi_offset s0, -8
    addi    s0, sp, 32
    .cfi_def_cfa s0, 0
    sw  a0, -12(s0)
    sw  a1, -16(s0)
    sw  a2, -20(s0)
    li  a0, 0
    sw  a0, -24(s0)
    lw  a0, -12(s0)
    sw  a0, -28(s0)
    j   .LBB1_1
.LBB1_1:
    lw  a0, -28(s0)
    lw  a1, -16(s0)
    bge a0, a1, .LBB1_4
    j   .LBB1_2
.LBB1_2:
    lw  a1, -28(s0)
    lw  a0, -24(s0)
    add a0, a0, a1
    sw  a0, -24(s0)
    lw  a0, -24(s0)
    call    _Z3outi
    j   .LBB1_3
.LBB1_3:
    lw  a0, -28(s0)
    addi    a0, a0, 1
    sw  a0, -28(s0)
    j   .LBB1_1
.LBB1_4:
    lw  a0, -24(s0)
    lw  a1, -20(s0)
    sw  a0, 0(a1)
    lw  ra, 28(sp)
    lw  s0, 24(sp)
    addi    sp, sp, 32
    ret
.Lfunc_end1:
    .size   _Z4foo2iiPi, .Lfunc_end1-_Z4foo2iiPi
    .cfi_endproc

    .ident  "clang version 14.0.0 (https://github.com/tru/llvm-release-build fc075d7c96fe7c992dde351695a5d25fe084794a)"
    .section    ".note.GNU-stack","",@progbits
    .addrsig
    .addrsig_sym _Z3outi

