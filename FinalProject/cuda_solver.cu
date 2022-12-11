#include <cstring>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

#include "helper.cuh"

using namespace std;

// Kernel Fill empty cells for all the N2 boards in the sudoku array with the empty cells index

__global__ void fill_empty_cells(int* sudoku, int total_boards, int* empty_cells, int* num_empty_cells){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("id: %d\n", id);
    while(id < total_boards){
        int* board = sudoku + id * N2;
        int* empty_cells_ptr = empty_cells + id * N2;
        int empty_space_count = 0;
        for(int i = 0; i < N2; i++){
            if(board[i] == 0){
                empty_cells_ptr[empty_space_count] = i;
                empty_space_count++;
            }
        }
        num_empty_cells[id] = empty_space_count;
        id += blockDim.x * gridDim.x;
    }
}


// Iterative Backtracking kernel
__global__ void backtracking(int* sudoku, int total_boards, int* empty_cells, int num_empty_cells, int* solved, int* lock){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    while(id < total_boards && *solved == 0){
        int* board = sudoku + id * N2;
        int empty_id = 0;

        while(empty_id>=0 && empty_id < num_empty_cells && *solved == 0){
            int empty_space = empty_cells[empty_id];
            int row = empty_space / N;
            int col = empty_space % N;

            int num = board[empty_space] + 1;
            while(num <= N){
                if(check(board, row, col, num)){
                    board[empty_space] = num;
                    empty_id++;
                    break;
                }
                num++;
            }

            if(num > N){
                board[empty_space] = 0;
                empty_id--;
            }
        }
        if(empty_id == num_empty_cells && *solved == 0){

            // make sure that only one thread saves the solution using the atomicXchg
            if(atomicExch(lock, 1) == 0){
                if(*solved == 1){
                    atomicExch(lock, 0);
                    return;
                }
                
                printf("Solved!!!!!!!!!!!!!!!!\n");
                *solved = 1;
                memcpy(sudoku, board, N2*sizeof(int));
                __threadfence();
                atomicExch(lock, 0);
            }
        }
        id += blockDim.x * gridDim.x;
    }
}


// BFS kernel
__global__ void bfs(int* prev_sudoku, int* next_sudoku, int total_boards, int* boards_ptr, int *solved, int max_boards){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    while(id < total_boards){
        int* prev_board = prev_sudoku + id * N2;

        int empty_space;
        if (find_empty(prev_board, empty_space)){
            int row = empty_space / N;
            int col = empty_space % N;

            for(int i = 1; i <= N; i++) {
                if(check(prev_board, row, col, i)){
                    int next_boards_ptr = atomicAdd(boards_ptr, 1);
                    if(next_boards_ptr >= max_boards){
                        *solved = -1;
                        return;
                    }
                    // printf("id: %d, next_boards_ptr: %d, boards_ptr: %d\n", id, next_boards_ptr, *boards_ptr);
                    int* next_board = next_sudoku + next_boards_ptr * N2;
                    memcpy(next_board, prev_board, N2*sizeof(int));
                    // for(int j = 0; j < N2; j++){
                    //     next_board[j] = prev_board[j];
                    // }
                    next_board[empty_space] = i;
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
    // max_boards = 2^28
    // 2^2 * 81 = 324 size of one board
    // 324 * 2^28 = 6GB

    int* max_boards;
    cudaMallocManaged(&max_boards, sizeof(int));
    *max_boards= pow(2, 12);
    int tot_size_boards = N2 * (*max_boards);

    cudaMallocManaged(&prev_sudoku, tot_size_boards * sizeof(int));
    memset(prev_sudoku, 0, tot_size_boards * sizeof(int));
    memcpy(prev_sudoku, sudoku, N2*sizeof(int));

    cudaMallocManaged(&next_sudoku, tot_size_boards * sizeof(int));
    memset(next_sudoku, 0, tot_size_boards * sizeof(int));

    int* boards_ptr;
    cudaMallocManaged(&boards_ptr, sizeof(int));
    *boards_ptr = 0;

    int* total_boards;
    cudaMallocManaged(&total_boards, sizeof(int));
    *total_boards = 1;

    int* solved;
    cudaMallocManaged(&solved, sizeof(int));
    *solved = 0;

    int* empty_cells;
    cudaMallocManaged(&empty_cells, N2 * sizeof(int));
    memset(empty_cells, 0, N2 * sizeof(int));

    int* num_empty_cells;
    cudaMallocManaged(&num_empty_cells, sizeof(int));
    *num_empty_cells = 0;

    dim3 dimGrid(blocksPerGrid);
    dim3 dimBlock(threadsPerBlock);

    int iter = 0;
    int prev_total_boards = 0;
    
    while(*total_boards < (*max_boards)){
        bfs <<<dimGrid, dimBlock>>> (prev_sudoku, next_sudoku, *total_boards, boards_ptr, solved, *max_boards);
        cudaDeviceSynchronize();

        *total_boards = *boards_ptr;
        *boards_ptr = 0;

        if(*solved == 1){
            cout<<"Solved!!!!!!!"<<endl;
            memcpy(sudoku, prev_sudoku, N2*sizeof(int));
            print_sudoku(sudoku);
            write_file("result.txt", sudoku);
            return 0;
        }else if (*solved == -1){
            cout<<"Too many boards..."<<endl;
            cout<<"Starting backtracking..."<<endl;
            *total_boards = prev_total_boards;
            break;
        }

        if (*total_boards == 0){
            cout<<"No solution"<<endl;
            return 0;
        }
                
        cout<<"Iteration:"<<iter<<" Total boards:"<<*total_boards<<endl; 

        int *temp = prev_sudoku;
        prev_sudoku = next_sudoku;
        next_sudoku = temp;
        prev_total_boards = *total_boards;
        iter++;
    }


    cudaFree(next_sudoku);
    cout<<"Filling empty cells..."<<endl;
    int count = 0;
    for(int i=0; i<N2; i++){
        if(prev_sudoku[i] == 0){
            
            empty_cells[count] = i;
            count++;
        }
    }
    *num_empty_cells = count;


    cout<<"Backtracking..."<<endl;
    *solved = 0;
    int* lock;
    cudaMallocManaged(&lock, sizeof(int));
    *lock = 0;

    backtracking <<<dimGrid, dimBlock>>> (prev_sudoku, *total_boards, empty_cells, *num_empty_cells, solved, lock);
    cudaDeviceSynchronize();
    
    if(*solved==1){
        cout<<"Solution: "<<endl;
        memcpy(sudoku, prev_sudoku, N2*sizeof(int));
        print_sudoku(sudoku);
        write_file("result.txt", sudoku);
    }
    else{
        cout<<"No solution"<<endl;
    }
    
    cudaFree(prev_sudoku);
    cudaFree(boards_ptr);
    cudaFree(total_boards);
    cudaFree(solved);
    cudaFree(empty_cells);
    cudaFree(num_empty_cells);

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