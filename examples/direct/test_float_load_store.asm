# store and load floats

li x1, 123.456
sw x1, 100(x0)

li x1, 0.123
sw x1, 104(x0)

lw x2, 100(x0)
outr.s x2

lw x2, 104(x0)
outr.s x2

halt
