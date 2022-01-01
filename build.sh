#!/bin/bash
python3 main.py
nasm -felf64 output.asm
ld output.o -o output
