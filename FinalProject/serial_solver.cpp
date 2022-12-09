// sudoku solver

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>


using namespace std;


void read_file(string filename, vector<vector<int> > &sudoku){
    ifstream file(filename);
    string line;
    int lineCount = 0;
    while(getline(file, line) and lineCount<10){
        vector<int> row;
        lineCount+=1;
        if(lineCount==1){
            continue;
        }
        for(int i = 0; i < line.size(); i++){
            row.push_back(line[i] - '0');
        }
        sudoku.push_back(row);
    }
}

void print_sudoku(vector<vector<int> > &sudoku){
    for(int i = 0; i < sudoku.size(); i++){
        for(int j = 0; j < sudoku[i].size(); j++){
            cout << sudoku[i][j] << " ";
        }
        cout << endl;
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

bool solve(vector<vector<int> > &sudoku){
    int row, col;
    if(!find_empty(sudoku, row, col)){
        return true;
    }
    for(int i = 1; i <= 9; i++){
        if(check(sudoku, row, col, i)){
            sudoku[row][col] = i;
            if(solve(sudoku)){
                return true;
            }
            sudoku[row][col] = 0;
        }
    }
    return false;
}

int main(){
    vector<vector<int> > sudoku;
    read_file("sudoku.txt", sudoku);
    print_sudoku(sudoku);
    cout << endl;
    solve(sudoku);
    print_sudoku(sudoku);
    return 0;
}