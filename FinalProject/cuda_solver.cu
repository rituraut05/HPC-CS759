#include <cstring>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

#include "helper.h"

using namespace std;

int main(int argc, char* argv[]){
    if (argc != 4){
        cout<<"Usage: ./sudoku <threadsPerBlock> <blocksPerGrid> <filename>"<<endl;
        return 0;
    }

    int threadsPerBlock = atoi(argv[1]);
    int blocksPerGrid = atoi(argv[2]);
    string filename = argv[3];

    int* sudoku = new int[N2];
    read_file(filename, sudoku);
    print_sudoku(sudoku);
    cout<<endl;

    int* prev_sudoku = new int[N2];
    int* next_sudoku = new int[N2];
    int num_boards = pow(2, 5);
    int tot_size_boards = N2* num_boards;

    cudaMallocManaged(&prev_sudoku, tot_size_boards * sizeof(int));
    cudaMallocManaged(&next_sudoku, tot_size_boards * sizeof(int));
    memset(prev_sudoku, 0, tot_size_boards * sizeof(int));
    memset(next_sudoku, 0, tot_size_boards * sizeof(int));
    memcpy(prev_sudoku, sudoku, N2*sizeof(int));

    // print prev_sudoku
    for(int i = 0; i < tot_size_boards; i++){
        cout<<prev_sudoku[i]<<" ";
        if((i+1)%N == 0){
            cout<<endl;
        }
        if((i+1)%N2 == 0){
            cout<<endl;
        }
    }

    return 0;
}