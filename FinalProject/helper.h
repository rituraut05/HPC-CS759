#ifndef HELPER_H
#define HELPER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;

#define N 9
#define N2 81

void read_file(string filename, int* board);
void print_sudoku(int* board);
bool check_row(vector<vector<int> > &sudoku, int row, int num);
bool check_col(vector<vector<int> > &sudoku, int col, int num);
bool check_box(vector<vector<int> > &sudoku, int row, int col, int num);
bool check(vector<vector<int> > &sudoku, int row, int col, int num);
bool find_empty(vector<vector<int> > &sudoku, int &row, int &col);

#endif
