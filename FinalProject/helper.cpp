#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "helper.h"

void read_file(string filename, int* board){
    ifstream file(filename);
    string line;
    int i = 0;
    while(getline(file, line)){
        for(int j = 0; j < N; j++){
            board[i*N + j] = line[j] - '0';
        }
        i++;
    }
}

void print_sudoku(int* board){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout<<board[i*N + j]<<" ";
        }
        cout<<endl;
    }
}


bool check_row(vector<vector<int> > &sudoku, int row, int num){
    for(int i = 0; i < sudoku[row].size(); i++){
        if(sudoku[row][i] == num){
            return false;
        }
    }
    return true;
}

bool check_col(vector<vector<int> > &sudoku, int col, int num){
    for(int i = 0; i < sudoku.size(); i++){
        if(sudoku[i][col] == num){
            return false;
        }
    }
    return true;
}

bool check_box(vector<vector<int> > &sudoku, int row, int col, int num){
    int row_start = (row/3)*3;
    int col_start = (col/3)*3;
    for(int i = row_start; i < row_start+3; i++){
        for(int j = col_start; j < col_start+3; j++){
            if(sudoku[i][j] == num){
                return false;
            }
        }
    }
    return true;
}

bool check(vector<vector<int> > &sudoku, int row, int col, int num){
    return check_row(sudoku, row, num) and check_col(sudoku, col, num) and check_box(sudoku, row, col, num);
}

bool find_empty(vector<vector<int> > &sudoku, int &row, int &col){
    for(row = 0; row < sudoku.size(); row++){
        for(col = 0; col < sudoku[row].size(); col++){
            if(sudoku[row][col] == 0){
                return true;
            }
        }
    }
    return false;
}