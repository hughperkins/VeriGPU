# first make sure li is working correctly, then check li on floats

li x1, 123
outr x1
li x1, 2000
outr x1

addi x1, x0, 1000
outr x1
# this will produce a very postive output, since we are printing unsigned
addi x1, x0, -1000
outr x1
addi x1, x0, 2000
outr x1
# this will produce a very postive output, since we are printing unsigned
addi x1, x0, -2000
outr x1

# int(math.pow(2,24)) + 123 :
li x1, 16777339
outr x1
li x1, 12345678
outr x1
li x1, 123456789
outr x1
li x1, 1234567891
outr x1
li x1, 2345678912
outr x1
li x1, 3456789123
outr x1

li x1, -2.5
outr.s x1
li x1, 1.23
outr.s x1
li x1 123.456
outr.s x1
li x1 0.000123456
outr.s x1
halt
