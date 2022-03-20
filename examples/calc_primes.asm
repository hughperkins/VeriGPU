; calculate prime numbers up to 31
; we will do this the non-sieve way in this examples

li x1, 2  ; current candidate prime

check_prime:
    addi x1, x1, 1 ; go to next prime
    li x2, 2      ; check divisors
    inner_loop:
        remu x3, x1, x2  ; if this is zero, its not prime
        beq x3, x0, check_prime
        addi x2, x2, 1  ; if x2 equals x1, then it's prime, we are done
        bne x2, x1, inner_loop
        ; if we get here, we are prime, so output yet
        outr x1
        ; if x1 is less than 30, try another one
        li x4, 31
        bne x1, x4, check_prime
halt
