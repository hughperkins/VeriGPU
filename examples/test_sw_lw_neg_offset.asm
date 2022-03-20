# negative offsets for load and store

li x1, 111
sw x1, 200(x0)

li x1, 222
sw x1, 300(x0)

li x3, 500

lw x1, -300(x3)
outr x1

lw x1, -200(x3)
outr x1

li x1, 444
sw x1, -300(x3)

li x1, 555
sw x1, -200(x3)

lw x1, 200(x0)
outr x1

lw x1, 300(x0)
outr x1

halt
