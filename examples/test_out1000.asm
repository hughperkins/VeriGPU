# test writing to 1000 as stdout

addi x31, x0, 1000

li x1, 123
sw x1, 0(x31)

li x1, 44
sw x1, 0(x31)

li x1, 55
sw x1, 0(x31)

halt
