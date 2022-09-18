#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -c 2
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err

for i in {10..30}
do
    ./task1 $((2**$i))
done
