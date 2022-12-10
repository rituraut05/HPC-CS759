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


bool check_row(int* &sudoku, int row, int num){
    for(int i = 0; i < N; i++){
        if(sudoku[row*N + i] == num){
            return false;
        }
    }
    return true;
}

bool check_col(int* &sudoku, int col, int num){
    for(int i = 0; i < N; i++){
        if(sudoku[i*N + col] == num){
            return false;
        }
    }
    return true;
}

bool check_box(int* &sudoku, int row, int col, int num){

    int row_start = (row/Nr)*Nr;
    int col_start = (col/Nr)*Nr;
    for(int i = 0; i < Nr; i++){
        for(int j = 0; j < Nr; j++){
            if(sudoku[(row_start+i)*N + col_start+j] == num){
                return false;
            }
        }
    }
    return true;
}

bool check(int* &sudoku, int row, int col, int num){
    return check_row(sudoku, row, num) and check_col(sudoku, col, num) and check_box(sudoku, row, col, num);
}

bool find_empty(int* &sudoku, int &row, int &col){
    for(row = 0; row < N; row++){
        for(col = 0; col < N; col++){
            if(sudoku[row*N + col] == 0){
                return true;
            }
        }
    }
    return false;
}