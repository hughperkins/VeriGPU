# calculate 5!

li x1 5
li x2 1
li x3 1

loop:
    mul x2, x2, x1
    addi x1, x1, -1
    bne x1, x3, loop

outr x2
halt
