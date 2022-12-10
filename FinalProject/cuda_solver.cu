#include <cstring>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

#include "helper.cuh"

using namespace std;

// BFS kernel
__global__ void bfs(int* prev_sudoku, int* next_sudoku, int total_boards, int* boards_ptr, int* empty_cells, int* num_empty_cells, int *solved){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    while(id < total_boards){
        int* prev_board = prev_sudoku + id * N2;

        int* empty_cells_ptr = empty_cells + id * N2;

        int empty_space;
        if (find_empty(prev_board, empty_space)){
            int row = empty_space / N;
            int col = empty_space % N;

            for(int i = 1; i <= N; i++){
                if(check(prev_board, row, col, i)){
                    int next_boards_ptr = atomicAdd(boards_ptr, 1);
                    printf("id: %d, next_boards_ptr: %d, boards_ptr: %d\n", id, next_boards_ptr, *boards_ptr);
                    int* next_board = next_sudoku + next_boards_ptr * N2;
                    memcpy(next_board, prev_board, N2*sizeof(int));
                    next_board[empty_space] = i;

                    int empty_space_count = 0;
                    for(int i = 0; i < N2; i++){
                        if(next_board[i] == 0){
                            empty_cells_ptr[empty_space_count] = i;
                            empty_space_count++;
                        }
                    }
                    num_empty_cells[next_boards_ptr] = empty_space_count - 1;
                }
            }
        }
        else{
            *solved = 1;
            return;
        }
        id += blockDim.x * gridDim.x;
    }
} 
 


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

    // 6GB total memory to be used
    // num_boards = 2^28
    // 2^2 * 81 = 324 size of one board
    // 324 * 2^28 = 6GB

    cout<<"Assigning memory..."<<endl;
    int num_boards = pow(2, 20);
    int tot_size_boards = N2 * num_boards;

    cout<<"Assigning memory... to prev_sudoku"<<endl;
    cudaMallocManaged(&prev_sudoku, tot_size_boards * sizeof(int));
    memset(prev_sudoku, 0, tot_size_boards * sizeof(int));
    memcpy(prev_sudoku, sudoku, N2*sizeof(int));

    cout<<"Assigning memory... to next_sudoku"<<endl;
    cudaMallocManaged(&next_sudoku, tot_size_boards * sizeof(int));
    memset(next_sudoku, 0, tot_size_boards * sizeof(int));

    int* empty_cells;
    cout<<"Assigning memory... to empty_cells"<<endl;
    cudaMallocManaged(&empty_cells, tot_size_boards * sizeof(int));
    memset(empty_cells, 0, tot_size_boards * sizeof(int));

    int* num_empty_cells;
    cout<<"Assigning memory... to num_empty_cells"<<endl;
    cudaMallocManaged(&num_empty_cells, num_boards * sizeof(int));
    memset(num_empty_cells, 0, num_boards * sizeof(int));

    int* boards_ptr;
    cout<<"Assigning memory... to boards_ptr"<<endl;
    cudaMallocManaged(&boards_ptr, sizeof(int));
    *boards_ptr = 0;

    int* total_boards;
    cout<<"Assigning memory... to total_boards"<<endl;
    cudaMallocManaged(&total_boards, sizeof(int));
    *total_boards = 1;

    int* solved;
    cudaMallocManaged(&solved, sizeof(int));
    *solved = 0;

    dim3 dimGrid(blocksPerGrid);
    dim3 dimBlock(threadsPerBlock);

    int iter = 0;
    while(iter<=100){
        bfs <<<dimGrid, dimBlock>>> (prev_sudoku, next_sudoku, *total_boards, boards_ptr, empty_cells, num_empty_cells, solved);
        cudaDeviceSynchronize();

        if(*solved == 1){
            cout<<"Solved!!!!!!!"<<endl;
            break;
        }
        *total_boards = *boards_ptr;
        *boards_ptr = 0;

        cout<<*total_boards<<endl;

        cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Iteration: "<<iter<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl; 

        int *temp = prev_sudoku;
        prev_sudoku = next_sudoku;
        next_sudoku = temp;
        iter++;
        cout<<"#############################################"<<endl;
        cout<<"Prev sudoku"<<endl;
        for(int i = 0; i < *total_boards * N2; i++){
            cout<<prev_sudoku[i]<<" ";
            if((i+1)%N == 0){
                cout<<endl;
            }
            if((i+1)%N2 == 0){
                cout<<endl;
            }
        }

        cout<<"#############################################"<<endl;
        cout<<"Next sudoku"<<endl;
        for(int i = 0; i < *total_boards * N2; i++){
            cout<<next_sudoku[i]<<" ";
            if((i+1)%N == 0){
                cout<<endl;
            }
            if((i+1)%N2 == 0){
                cout<<endl;
            }
        }

    }
    
    return 0;
}


__device__ bool check_row(int* &sudoku, int row, int num){
    for(int i = 0; i < N; i++){
        if(sudoku[row*N + i] == num){
            return false;
        }
    }
    return true;
}

__device__ bool check_col(int* &sudoku, int col, int num){
    for(int i = 0; i < N; i++){
        if(sudoku[i*N + col] == num){
            return false;
        }
    }
    return true;
}

__device__ bool check_box(int* &sudoku, int row, int col, int num){

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

__device__ bool check(int* &sudoku, int row, int col, int num){
    return check_row(sudoku, row, num) && check_col(sudoku, col, num) && check_box(sudoku, row, col, num);
}

__device__ bool find_empty(int* &sudoku, int &index){
    for(int i = 0; i < N2; i++){
        if(sudoku[i] == 0){
            index = i;
            return true;
        }
    }
    return false;
}