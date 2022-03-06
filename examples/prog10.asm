addi x31, x0, 1000

li x1, 123
li x2, 100
sw x1, 0(x2)

li x1, 222
li x2, 110
sw x1, 0(x2)

li x2, 100
lw x3, 0(x2)
sw x3, 0(x31)

li x2, 110
lw x3, 0(x2)
sw x3, 0(x31)

halt
