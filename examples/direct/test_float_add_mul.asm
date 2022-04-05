; float addition

li x1, 1.234
li x2, 5.000
fadd.s x3, x1, x2
outr.s x3

li x1, 1.234
li x2, -5.000
fadd.s x3, x1, x2
outr.s x3

li x1, 1234.0
li x2, -5000.0
fadd.s x3, x1, x2
outr.s x3

li x1, 1000.0
li x2, -0.5
fadd.s x3, x1, x2
outr.s x3

; mul

li x1, 1.234
li x2, 5.000
fmul.s x3, x1, x2
outr.s x3 ; 6.17

li x1, 1.234
li x2, -5.000
fmul.s x3, x1, x2
outr.s x3  ; -6.17

li x1, 1234.0
li x2, -5000.0
fmul.s x3, x1, x2
outr.s x3 ; 6170000

li x1, 1000.0
li x2, -0.5
fmul.s x3, x1, x2
outr.s x3  ; -500

halt
