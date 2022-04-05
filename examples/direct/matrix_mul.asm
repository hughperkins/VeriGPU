; do a matrix multiplication, of two real matrices
li x29, data   ; x29 is current address
lw x28, 0 x29  ; x28 is number of multiplicaitons to go
li x24, results  ; results area

outr x28

addi x29, x29, 4   ; x29 is now start of matrix size declarations

loop_mmul:   ; at this point, x29 should be at start of multiplication declarations
lw x20, 0, x29 ; I = x20
lw x21, 4, x29 ; K = x21
lw x22, 8, x29 ; J = x22
outr x20
outr x21
outr x22
addi x29, x29, 12   ; x29 is now start of first matrix

mv x25, x29  ;       x25 is address of first matrix
mul x10, x20 x21 ;   temp x10 is size of first matrix, in words
slli x10, x10, 2   ; x10 is size of first mstrix, in bytes
add x26, x29, x10  ; x26 is address of second matrix

mul x10, x21, x22  ; size of second matrix, in words
slli x10, x10, 2   ; x10 is siz of second matrix, in bytes
add x27, x26, x10  ; x27 is address of next matrix mult declaration


li x10 0  ; i
li x11 0  ; k
li x12 0  ; j
; x28 number of multiplications to go, including this one
; x25 first matrix
; x26 second matrix
; x27 result matrix
; i/I is rows of first matrix
; j/J is cols of second matrix
; k/K is cols of first and rows of second
; linear index into first is i * K + k
; linear index into second is k * J + j
; linear index into result is i * J * j
; x20 I
; x21 K
; x22 J
; x10 i
; x11 k
; x12 j
; x5  sum over k, for single result output
; x2  value from first matrix
; x3  value ffrom second matrix
; x4  product of two matrix values

; not trying any kind of optimization here, just doing in ijk order...
; (here k is the middle dimension)
; for each i,j, we have to loop over k, multiplying the two corresponding values
; and ... store the result
; after storing the result, then print it out
i_loop:
    li x12, 0
    j_loop:
        li x11, 0
        li x5, 0.0
        k_loop:

            ; first matrix
            ; outr x11
            ; linear index into first is i * K + k
            ;                           = x10 * x21 + x11
            mul x6, x10, x21  ; number of words from row
            add x6, x6, x11   ; number o fwords including column
            slli x6, x6, 2    ; linear index in bytes
            add x6, x6, x25    ; absolute memory address of byte
            lw x2, 0, x6    ; get value from first matrix
            outr.s x2

            ; second matrix
            ; linear index into second is k * J + j
                                        ; = x11 * x22 + x12
            mul x6, x11, x22 ;  number of words from row
            add x6, x6, x12    ; number of words including column
            slli x6, x6, 2    ; number of bytes
            add x6, x6, x26  ; abs addr of byte
            lw x3, 0, x6
            outr.s x3

            fmul.s x4, x2, x3    ; product of two matrix values
            outr.s x4
            fadd.s x5, x5, x4    ; add to sum over k

            addi x11, x11, 1
            bne x11, x21, k_loop
        ; now we've summed over k, write the result to result matrix
        ; linear index into result is i * J * j
                                ;   x10 * x22 + x12
        mul x6, x10, x22
        add x6, x6, x12
        slli x6, x6, 2    ; in bytes
        add x6, x6, x24  ; absolute address
        sw x5, 0, x6      ; store result

        addi x12, x12, 1

        bne x12, x22, j_loop

    addi x10, x10, 1
    bne x10, x20, i_loop


; now write out the result
li x10 0  ; i
li x12 0  ; j
out 999 ; so we can see the start of matrix result
out 999 ; so we can see the start of matrix result
out 999 ; so we can see the start of matrix result
out 999 ; so we can see the start of matrix result
disp_i_loop:
    li x12, 0
    disp_j_loop:
        ; linear index into result is i * J * j
                                ;   x10 * x22 + x12
        mul x6, x10, x22
        add x6, x6, x12
        slli x6, x6, 2    ; in bytes
        add x6, x6, x24  ; absolute address
        lw x5, 0, x6
        outr.s x5

        addi x12, x12, 1
        bne x12, x22, disp_j_loop
    addi x10, x10, 1
    bne x10, x20, disp_i_loop

addi x29, x27, 0
addi x28, x28, -1
bne x28, x0, loop_mmul

finish:
halt

data:

word 2  ; number of multiplications
word 1  ; rows in first matrix
word 1  ; cols in first, and rows in second matrix
word 1  ; cols in second matrix
word 1.2
word 3.5

; result will be 4.2

word 2
word 3
word 4

; first matrix 2x3
; 1.7 2.4 5.5
; 4.1 1.3 4.4

; 1.7 * 3.3 + 2.4 * -10.0 + 5.5 * 15.2   1.7 * 9.1 + 2.4 * -2.5 + 5.5 * 3.2    1.7 * 2.8 + 2.4 * 15.0 + 5.5 * 3.1
;                                                                                 1.7 * -0.1 2.4 * 8.3 5.5 * 8.8
; 4.1 * 3.3 + 1.3 * -10.0 + 4.4 * 15.2   4.1 * 9.1 + 1.3 * -2.5 + 4.4 * 3.2    4.1 * 2.8 + 1.3 * 15.0 + 4.4 * 3.1 
;                                                                                  4.1 * -0.1 + 1.3 * 8.3 + 4.4 * 8.8
; 

word 1.7
word 2.4
word 5.5

word 4.1
word 1.3
word 4.4

; second matrix 3 x 4
;   3.3  9.1  2.8 -0.1
; -10.0 -2.5 15.0  8.3
;  15.2  3.2  3.1  8.8
word 3.3
word 9.1
word 2.8
word -0.1

word -10.0
word -2.5
word 15.0
word 8.3

word 15.2
word 3.2
word 3.1
word 8.8

results:

; check what the results should be, using pytorch:
; In [1]: import torch

; In [2]: a = torch.Tensor([[1.7,2.4,5.5],[4.1,1.3,4.4]])

; In [3]: b= torch.Tensor([[3.3, 9.1, 2.8, -0.1],[-10.0, -2.5, 15.0, 8.3], [15.2,3.2,3.1,
;    ...: 8.8]])

; In [4]: a @ b
; Out[4]: 
; tensor([[65.2100, 27.0700, 57.8100, 68.1500],
;         [67.4100, 48.1400, 44.6200, 49.1000]])

