; test outloc

outloc 0xf0; 240
outloc 0xf4
outloc 0xf0
outloc 0xf4
outloc 256
outloc 260
outloc 264
outloc 268
halt

location 0xf0:
word 0x0000dead
word 0x0000beef

location 256:
word 0x00001111
word 0x00002222
word 15
word 21
