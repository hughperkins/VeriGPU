; mul

li x1, 15
li x2, 31
mul x3, x1, x2
outr x3  ; 465

li x1, 0
li x2, 31
mul x3, x1, x2
outr x3  ; 0

li x1, 15
li x2, 0
mul x3, x1, x2
outr x3  ; 0

li x1, 347911
li x2, 12345
mul x3, x1, x2
outr x3  ; 4294961295

; divu, remu

li x1, 4
li x2, 4
divu x3, x1, x2
remu x4, x1, x2
outr x3  ; 1 r 0
outr x4

li x1, 4
li x2, 2
divu x3, x1, x2
remu x4, x1, x2
outr x3  ; 2 r 0
outr x4

li x1, 0
li x2, 0
divu x3, x1, x2
remu x4, x1, x2
outr x3  ; ffffffff r 0
outr x4

li x1, 0
li x2, 2
divu x3, x1, x2
remu x4, x1, x2
outr x3  ; 0 r 0
outr x4

li x1, 2222
li x2, 123
divu x3, x1, x2
remu x4, x1, x2
outr x3  ; 18 r 8
outr x4

li x1, 1111
li x2, 444
divu x3, x1, x2
remu x4, x1, x2
outr x3  ; 2 r 223
outr x4

halt
