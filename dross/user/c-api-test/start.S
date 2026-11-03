/* The kernel passes the bootstrap handle in rdi. Establish the C stack ABI. */
.text
.globl _start
.type _start,@function
_start:
    xorq %rbp, %rbp
    andq $-16, %rsp
    call c_main
    ud2
.size _start, .-_start
.section .note.GNU-stack,"",@progbits
