; sieve of eratosthenes, for prime numbers up to 31
; in this version, since we can currenlty only write words, we
; are going to use an entire word to mark a prime or not

; first we have to initialize the memory
; then run the sieve
; then write out the results

li x1, 200  ; base for our memory
li x2, 31   ; max prime

; initialize memory
li x3, 0
li x4, 1
addi x6, x2, 1  ; for loop termination
init_loop:
    ; shift left twice, and add base
    addi x5, x3, 0
    slli x5, x5, 2
    add x5, x1, x5
    sw x4, 0, x5
    addi x3, x3, 1
    bne x6, x3, init_loop

; do the sieve
li x3, 2  ; stride size
sieve_outer:
    addi x4, x3, 0  ; x4 = x3
    add x4, x4, x3   ; x4 += x3
    sieve_inner:
        ; shift left twice, and add base
        addi x5, x4, 0
        slli x5, x5, 2
        add x5, x1, x5
        sw x0, 0, x5  ; mark as not prime
        add x4, x4, x3  ; x4 += x3
        ble x4, x2, sieve_inner   ; if x4 < x2 keep going
    addi x3, x3, 1  ; x3 += 1
    bne x3, x2, sieve_outer  ; if x3 < 31, run seive

; write out the results
li x3, 1
write_loop:
    addi x3, x3, 1  ; x3 += 1
    ; shift left twice, and add base
    addi x5, x3, 0
    slli x5, x5, 2
    add x5, x1, x5
    lw x4, 0, x5

    beq x4, x0, after_prime_print
        outr x3
    after_prime_print:
    blt x3, x2, write_loop
end:

halt
