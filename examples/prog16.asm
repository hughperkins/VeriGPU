# sum numbers from 0 to 5

li x1, 5
li x2, 0

outr x2

loop:
    outr x1
    add x2, x1, x2
    addi x1, x1, -1
    bne x1, x0, loop

outr x1
outr x2

halt
