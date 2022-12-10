#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J project
#SBATCH -o project.out -e project.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00

./cuda 32 1 ./puzzles/easy_1.txt