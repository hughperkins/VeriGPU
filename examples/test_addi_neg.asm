# negative immediate for addi

li x1, 123
outr x1

addi x3, x1, 2
outr x3

addi x3, x1, -1
outr x3

addi x3, x1, -10
outr x3

addi x3, x1, -100
outr x3

addi x3, x0, -1000
outr x3

addi x3, x0, -2000
outr x3

addi x3, x0, 2000
outr x3

halt
