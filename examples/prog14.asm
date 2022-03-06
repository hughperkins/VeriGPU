# branch

li x1, 5

loop:
    outr x1
    addi x1, x1, -1
    bne x1, x0, loop

outr x1

halt
