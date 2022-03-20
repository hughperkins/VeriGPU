# test writing to 1000 as stdout

addi x31, x0, 1000

li x1, 123
sw x1, 0(x31)

li x1, 44
sw x1, 0(x31)

li x1, 55
sw x1, 0(x31)

# test out

out 123
out 789
out 0x32
out 0x44
out 50
out 60

; test outloc

outloc 300; 240
outloc 304
outloc 300
outloc 304
outloc 356
outloc 360
outloc 364
outloc 368

halt

# test halt, by putting instructions after it

out 70
out 80


location 300:
word 0x0000dead
word 0x0000beef

location 356:
word 0x00001111
word 0x00002222
word 15
word 21
