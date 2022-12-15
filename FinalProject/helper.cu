#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "helper.cuh"
#include "cuda_runtime.h"


void read_file(string filename, int* board){
    ifstream file(filename);
    string line;
    int i = 0;
    while(getline(file, line)){
        int j = 0;
        int num = 0;
        for(int k = 0; k < line.length(); k++){
            if(line[k] == ' '){
                board[i*N + j] = num;
                j++;
                num = 0;
            }
            else{
                num = num*10 + (line[k] - '0');
            }
        }
        board[i*N + j] = num;
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

void write_file(string filename, int* board){
    ofstream file(filename);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            file<<board[i*N + j];
        }
        file<<endl;
    }
}