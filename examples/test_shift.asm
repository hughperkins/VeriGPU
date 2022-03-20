# try add, sll, srl

li x1 15
li x2 8
add x3, x1, x2
outr x3

li x2 1
sll x3, x1, x2
outr x3

li x2 2
sll x3, x1, x2
outr x3

li x2 3
sll x3, x1, x2
outr x3

li x2 1
srl x3, x1, x2
outr x3

li x2 2
srl x3, x1, x2
outr x3

li x2 3
srl x3, x1, x2
outr x3

halt
