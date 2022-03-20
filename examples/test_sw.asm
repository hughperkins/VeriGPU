# test sw

li x2, 0xf0
li x1, 123
outr x1
outr x2
sw x1, 0(x2)
outloc 0xf0

li x2, 0xf4
li x1, 111
sw x1, 0(x2)
outloc 0xf4

outloc 0xf0
outloc 0xf4

halt
