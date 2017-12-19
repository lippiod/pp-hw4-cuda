#!/bin/bash

if [ -z "$2" ]; then
    echo "1: testcase"
    echo "2: phase (0 1 2 3)"
    echo "3: block factor (optinal, default 32)"
    echo "Too few arguments."
    exit
fi

if [ "$1" == "rd" ]; then
    IN_FILE="$1.in"
else
    IN_FILE="testcase/$1.in"
fi
OUTFILE="$1.out"
ANSFILE="testcase/$1.ans"
SEQFILE="${1}s.out"

OUTTXT="${OUTFILE}.txt"
ANSTXT="$1.ans.txt"
SEQTXT="${SEQFILE}.txt"

BFACTOR="32"
if [ -n "$3" ]; then
    BFACTOR="$3"
fi

srun -ppp -n 1 --gres=gpu:1 ./HW4_cuda $IN_FILE $OUTFILE $BFACTOR

SEQPRE="hw4_block_FW"
if [ "$2" -ne "3" ]; then
    ANSFILE="$SEQFILE"
    ANSTXT="$SEQTXT"
    ${SEQPRE}_p$2 $IN_FILE $ANSFILE $BFACTOR
fi

cmp $OUTFILE $ANSFILE
if [ $? -ne 0 ]; then
    echo "input vertex number:"
    read V
    hw4_b2txt $V $OUTFILE $OUTTXT
    hw4_b2txt $V $ANSFILE $ANSTXT
fi
