.section .text
.global _start

_start:
    # Initialize registers
    li a0, 5          # Example: n = 5
    li t1, 1          # Counter i = 1
    li t2, 0          # Sum = 0

loop:
    # Check if counter > n
    bgt t1, a0, done

    # Add current number to sum
    add t2, t2, t1    # sum += i
    addi t1, t1, 1    # i++
    j loop

done:
    # Exit program
    mv a0, t2         # Move result to a0
    li a7, 93         # Exit syscall
    ecall

.section .data
