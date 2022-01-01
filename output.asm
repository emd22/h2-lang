
global _start

section .text
_start:
mov rsi, hw_str
mov rdx, 14
mov rdi, 1
mov rax, 1
syscall
mov rax, 60
syscall

section .data
hw_str: db 'Hello, World!', 10

section .bss

        