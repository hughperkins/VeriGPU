# test writing to 1000 as stdout

li x31, 1000000

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

outloc 600; 240
outloc 604
outloc 600
outloc 604
outloc 656
outloc 660
outloc 664
outloc 668

halt

# test halt, by putting instructions after it

out 70
out 80


location 600:
word 0x0000dead
word 0x0000beef

location 656:
word 0x00001111
word 0x00002222
word 15
word 21
