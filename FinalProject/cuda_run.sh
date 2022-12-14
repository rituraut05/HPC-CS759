#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J project
#SBATCH -o project.out -e project.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00

cuda-memcheck ./cuda 512 256 ./puzzles/16/easy1.txt
# nvidia-smi

