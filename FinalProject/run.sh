#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J project
#SBATCH -o project.out -e project.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00

nvcc cuda_solver.cu helper.cu -o cuda
./cuda 512 256 ./puzzles/9/easy_1.txt
