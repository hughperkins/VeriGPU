# LUI, AUIPC

lui x1, 0xabcde
outr x1
addi x1, x1, 0x321
outr x1

auipc x1, 0x111
outr x1

; LI
li x2, 0xabcde321
outr x2

li x2, 0xabcdefff
outr x2

li x2, 0xabcde000
outr x2

addi x1, x0, 10
outr x1

addi x1, x0, -10
outr x1

halt
