# test non-zero offset to load and store

li x1, 111
sw x1, 200(x0)

li x1, 222
sw x1, 210(x0)

lw x3, 200(x0)
outr x3
lw x3, 210(x0)
outr x3

li x2, 100
outr x2

li x3, 116
outr x3

li x1, 444
sw x1, 200(x2)

li x1, 555
sw x1, 210(x3)

lw x4, 200(x2)
outr x4
lw x4, 210(x3)
outr x4

lw x4, 184(x3)
outr x4
lw x4, 226(x2)
outr x4

halt
