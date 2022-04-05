li x2, 3
li x3, 5

slti x1, x2, 5
outr x1

slti x1, x3, 3
outr x1

slti x1, x3, 2
outr x1

li x2, 1

andi x1, x2, 0
outr x1
andi x1, x2, 1
outr x1
andi x1, x0, 0
outr x1
andi x1, x0, 1
outr x1

li x2, 5
outr x2

slli x1, x2, 0
outr x1

slli x1, x2, 1
outr x1

slli x1, x2, 2
outr x1

slli x1, x2, 3
outr x1


li x2, 73
outr x2

srli x1, x2, 0
outr x1

srli x1, x2, 1
outr x1

srli x1, x2, 2
outr x1

srli x1, x2, 3
outr x1

halt
