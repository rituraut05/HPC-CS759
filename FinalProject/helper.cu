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

void write_file(string filename, int* board){
    ofstream file(filename);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            file<<board[i*N + j];
        }
        file<<endl;
    }
}