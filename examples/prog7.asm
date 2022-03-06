li x2, 0x40
li x1, 123
outr x1
outr x2
sw x1, 0(x2)
outloc 0x40

li x2, 0x54
li x1, 111
sw x1, 0(x2)
outloc 0x54

outloc 0x40
outloc 0x54

halt
