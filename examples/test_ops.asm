li x2, 7
li x3, 11

slt x1, x2, x3
outr x1  ; should be 1

slt x1, x3, x2
outr x1  ; should be 0

slt x1, x2, x2
outr x1  ; should be 0

sltu x1, x0, x2
outr x1  ; should be 1

li x2, 0
sltu x1, x0, x2
outr x1  ; should be 0

li x2, -1
sltu x1, x0, x2
outr x1  ; should be 1

li x2, 0b1011
li x3, 0b0110
and x1, x2, x3
outr x1  ; 0b0010

halt
