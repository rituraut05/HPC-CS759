#!/usr/bin/env bash

# a) Change the current directory to a subdirectory called somedir
cd somedir

# b) Print out to the terminal the contents of a file called sometext.txt.
cat sometext.txt

# c) Print out to the terminal the last 5 lines of sometext.txt.
tail -5 sometext.txt

# d) Print out to the terminal the last 5 lines of each file that ends in the extension .txt
for i in *.txt; do
    tail -5 "$i"
done

# e) Write a for loop which prints each integer from 0 to 6
for i in 0 1 2 3 4 5
do
    echo "$i"
done
