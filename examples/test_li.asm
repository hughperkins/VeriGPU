# test li with different registers, using outr
# also test outr

li x1 3
outr x1
li x1 27
outr x1
li x1 123
outr x1
li x1 444
outr x1
li x21 222
li x23 333
outr x21
outr x23

li x1 7
li x2 35
outr x1
outr x2
li x2 22
li x1 47
outr x1
outr x2

outr x1
outr x2

halt
