#ifndef HELPER_CUH
#define HELPER_CUH

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;

#define N 9
#define N2 81
#define Nr 3

void read_file(string filename, int* board);
void print_sudoku(int* board);
void write_file(string filename, int* board);
__device__ bool check_row(int* &sudoku, int row, int num);
__device__ bool check_col(int* &sudoku, int col, int num);
__device__ bool check_box(int* &sudoku, int row, int col, int num);
__device__ bool check(int* &sudoku, int row, int col, int num);
__device__ bool find_empty(int* &sudoku, int &index);

#endif
