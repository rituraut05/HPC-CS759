#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J omp_sudoku
#SBATCH -o %x.out -e %x.err
#SBATCH -N 1 -c 20
#SBATCH -t 0-00:30:00

./sudoku 2 puzzles/9/easy_1.txt


