# test li with different registers, using outr
# (also test outr)

li x1 0x01
outr x1
li x1 0x02
outr x1
li x1 0x1e
outr x1
li x1 0x2d
outr x1
li x21 22
li x23 33
outr x21
outr x23
halt
