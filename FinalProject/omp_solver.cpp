// write an openmp version of sudoku solver

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;

#define N 9
#define N2 81
#define Nr 3

struct board {
    int* matrix;
    uint64_t* possible;
} typedef board_t;


void read_file(string filename, int* board);
void print_sudoku(int* board);
void write_file(string filename, int* board);
bool check_row(int* &matrix, int row, int num);
bool check_col(int* &matrix, int col, int num);
bool check_box(int* &matrix, int row, int col, int num);
bool check(int* &matrix, int row, int col, int num);
bool find_empty(int* &matrix, int &index);
bool solve_partially(int* &matrix);
bool print_possible_values(board_t* board, int index);
bool lone_ranger_col(board_t* board);
bool lone_ranger_row(board_t* board);
bool lone_ranger_box(board_t* board);
bool twins_row(board_t* board);
bool twins_col(board_t* board);
bool twins_box(board_t* board);
bool twins(board_t* board);
bool update_neighbors(board_t* board, int index);



void read_file(string filename, int* board){
    ifstream file(filename);
    string line;
    int i = 0;
    while(getline(file, line)){
        for(int j = 0; j < N; j++){
            if (line[j] == ' '){
                continue;
            }
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


bool check_row(int* &matrix, int row, int num){
    for(int i = 0; i < N; i++){
        if(matrix[row*N + i] == num){
            return false;
        }
    }
    return true;
}

bool check_col(int* &matrix, int col, int num){
    for(int i = 0; i < N; i++){
        if(matrix[i*N + col] == num){
            return false;
        }
    }
    return true;
}

bool check_box(int* &matrix, int row, int col, int num){

    int row_start = (row/Nr)*Nr;
    int col_start = (col/Nr)*Nr;
    for(int i = 0; i < Nr; i++){
        for(int j = 0; j < Nr; j++){
            if(matrix[(row_start+i)*N + col_start+j] == num){
                return false;
            }
        }
    }
    return true;
}

bool check(int* &matrix, int row, int col, int num){
    return check_row(matrix, row, num) && check_col(matrix, col, num) && check_box(matrix, row, col, num);
}

bool find_empty(int* &matrix, int &index){
    for(int i = 0; i < N2; i++){
        if(matrix[i] == 0){
            index = i;
            return true;
        }
    }
    return false;
}

bool possible_values_in_cell(board_t* board, int index){
    int row = index/N;
    int col = index%N;
    uint64_t mask = 0x1;
    for(int i = 0; i < N; i++){
        if(check(board->matrix, row, col, i+1)){
            board->possible[index] |= mask;
        }
        mask <<= 1;
    }
    return true;
}

bool print_all_possible_values(board_t* board){
    for(int i = 0; i < N2; i++){
        if(board->matrix[i] != 0){
            cout<<board->matrix[i]<<" -- ";
        }

        print_possible_values(board, i);
    }
    return true;
}
bool print_possible_values(board_t* board, int index){
    uint64_t mask = 0x1;
    for(int i = 0; i < N; i++){
        if(board->possible[index] & mask){
            cout<<i+1<<" ";
        }
        mask <<= 1;
    }
    cout<<" -- ";
    return true;
}

bool possible_values(board_t* board){
    for(int i = 0; i < N2; i++){
        if(board->matrix[i] == 0){
            possible_values_in_cell(board, i);
            if (board->possible[i] == 0){
                return false;
            }
        }
    }
    return true;
}

bool update_neighbors(board_t* board, int index){
    int row = index/N;
    int col = index%N;
    uint64_t mask = board->possible[index];
    for(int i = 0; i < N; i++){
        if(board->matrix[row*N + i] == 0){
            board->possible[row*N + i] &= ~mask;
        }
        if(board->matrix[i*N + col] == 0){
            board->possible[i*N + col] &= ~mask;
        }
    }
    int row_start = (row/Nr)*Nr;
    int col_start = (col/Nr)*Nr;
    for(int i = 0; i < Nr; i++){
        for(int j = 0; j < Nr; j++){
            if(board->matrix[(row_start+i)*N + col_start+j] == 0){
                board->possible[(row_start+i)*N + col_start+j] &= ~mask;
            }
        }
    }
    return true;
}

bool lone_ranger_row(board_t* board){
    cout<<"lone ranger row"<<endl;
    bool flag = true;
    bool changed = false;
    while(flag){
        flag = false;
        for(int i = 0; i < N; i++){
            for (int k = 0; k < N; k++){
                int count = 0;
                int ind = 0;
                uint64_t mask = 1 << k;
                for(int j = 0; j < N; j++){
                    if(board->matrix[i*N+j]==0 && board->possible[i*N + j] & mask){
                        count++;
                        ind = j;
                        if(count > 1){
                            break;
                        }
                    }
                }
                if(count == 1){
                    board->matrix[i*N + ind] = k+1;
                    board->possible[i*N + ind] = 1 << k;
                    update_neighbors(board, i*N + ind);
                    flag = true;
                    changed = true;
                }
            }
        }
    }

    return changed;
}

bool lone_ranger_col(board_t* board){
    bool flag = true;
    bool changed = false;
    while(flag){
        flag = false;
        for(int i = 0; i < N; i++){
            for (int k = 0; k < N; k++){
                int count = 0;
                int ind = 0;
                uint64_t mask = 1 << k;
                for(int j = 0; j < N; j++){
                    if(board->matrix[j*N + i] == 0 && board->possible[j*N + i] & mask){
                        count++;
                        ind = j;
                        if(count > 1){
                            break;
                        }
                    }
                }
                if(count == 1){
                    board->matrix[ind*N + i] = k+1;
                    board->possible[ind*N + i] = 1 << k;
                    update_neighbors(board, ind*N + i);
                    flag = true;
                    changed = true;
                }
            }
        }
    }

    return changed;
}

bool lone_ranger_box(board_t* board){
    bool flag = true;
    bool changed = false;
    while(flag){
        flag = false;
        for(int i = 0; i < N; i += Nr){
            for(int j = 0; j < N; j += Nr){
                for (int k = 0; k < N; k++){
                    int count = 0;
                    int ind = 0;
                    uint64_t mask = 1 << k;
                    for(int p = 0; p < Nr; p++){
                        for(int q = 0; q < Nr; q++){
                            if(board->matrix[(i+p)*N + j+q] == 0 && board->possible[(i+p)*N + j+q] & mask){
                                count++;
                                ind = (i+p)*N + j+q;
                                if(count > 1){
                                    break;
                                }
                            }
                        }
                    }
                    if(count == 1){
                        board->matrix[ind] = k+1;
                        board->possible[ind] = 1 << k;
                        update_neighbors(board, ind);
                        flag = true;
                        changed = true;
                    }
                }
            }
        }
    }

    return changed;
}

bool twins_row(board_t* board){
    bool flag = true;
    bool changed = false;
    while(flag){
        flag = false;
        for(int i = 0; i < N; i++){
            for (int k = 0; k < N; k++){
                int count = 0;
                int ind1 = 0;
                int ind2 = 0;
                uint64_t mask = 1 << k;
                for(int j = 0; j < N; j++){
                    if(board->possible[i*N + j] & mask && board->matrix[i*N + j] == 0){
                        count++;
                        if(count == 1){
                            ind1 = j;
                        }
                        if(count == 2){
                            ind2 = j;
                        }
                        if(count > 2){
                            break;
                        }
                    }
                }
                if(count == 2){
                    for(int j = 0; j < N; j++){
                        if(j != ind1 && j != ind2){
                            board->possible[i*N + j] &= ~mask;
                            flag = true;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    return changed;
}

bool twins_col(board_t* board){
    bool flag = true;
    bool changed = false;
    while(flag){
        flag = false;
        for(int i = 0; i < N; i++){
            for (int k = 0; k < N; k++){
                int count = 0;
                int ind1 = 0;
                int ind2 = 0;
                uint64_t mask = 1 << k;
                for(int j = 0; j < N; j++){
                    if(board->possible[j*N + i] & mask && board->matrix[j*N + i] == 0){
                        count++;
                        if(count == 1){
                            ind1 = j;
                        }
                        if(count == 2){
                            ind2 = j;
                        }
                        if(count > 2){
                            break;
                        }
                    }
                }
                if(count == 2){
                    for(int j = 0; j < N; j++){
                        if(j != ind1 && j != ind2){
                            board->possible[j*N + i] &= ~mask;
                            flag = true;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    return changed;
}

bool twins_box(board_t* board){
    bool flag = true;
    bool changed = false;
    while(flag){
        flag = false;
        for(int i = 0; i < N; i += Nr){
            for(int j = 0; j < N; j += Nr){
                for (int k = 0; k < N; k++){
                    int count = 0;
                    int ind1 = 0;
                    int ind2 = 0;
                    uint64_t mask = 1 << k;
                    for(int p = 0; p < Nr; p++){
                        for(int q = 0; q < Nr; q++){
                            if(board->possible[(i+p)*N + j+q] & mask && board->matrix[(i+p)*N + j+q] == 0){
                                count++;
                                if(count == 1){
                                    ind1 = (i+p)*N + j+q;
                                }
                                if(count == 2){
                                    ind2 = (i+p)*N + j+q;
                                }
                                if(count > 2){
                                    break;
                                }
                            }
                        }
                    }
                    if(count == 2){
                        for(int p = 0; p < Nr; p++){
                            for(int q = 0; q < Nr; q++){
                                if((i+p)*N + j+q != ind1 && (i+p)*N + j+q != ind2){
                                    board->possible[(i+p)*N + j+q] &= ~mask;
                                    flag = true;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return changed;
}

bool eliminate(board_t* board){
    cout<<"eliminate"<<endl;
    bool flag = false;
    for(int i = 0; i < N2; i++){
        if(board->matrix[i] == 0 && __builtin_popcountll(board->possible[i] == 1)){
            board->matrix[i] = __builtin_ctzll(board->possible[i]) + 1;
            flag = true;
        }    
    }
    return flag;
}


board_t* init_board(int* matrix){
    board_t* board = new board_t;
    board->matrix = matrix;
    board->possible = new uint64_t[N2];
    for(int i = 0; i < N2; i++){
        if(matrix[i] == 0){
            board->possible[i] = 0;
        }
        else{
            board->possible[i] = 1 << (matrix[i] - 1);
        }
    }
    return board;
}

bool is_solved(board_t* board){
    for(int i = 0; i < N2; i++){
        if(board->matrix[i] == 0){
            return false;
        }
    }
    return true;
}

// int solve_partially(board_t* board){
    // possible_values(board);
    // int i = 0;
    // while(eliminate(board) || lone_ranger_row(board) || lone_ranger_col(board) || lone_ranger_box(board) || twins_row(board) || twins_col(board) || twins_box(board)){
    //     cout<<i++<<endl;
    //     if(is_solved(board)){
    //         return 1;
    //     }
    //     cout<<"hi"<<endl;
    //     if(!possible_values(board)){
    //         return -1;
    //     }
    //     cout<<"hello"<<endl;
    // }
    // return 0;
// }

int solve_partially(board_t* board){
    bool flag = true;
    while(flag){
        flag = false;
        possible_values(board);
        if(!eliminate(board)){
            if(!lone_ranger_row(board)){
                if(!lone_ranger_col(board)){
                    if(!lone_ranger_box(board)){
                        if(!twins_row(board)){
                            if(!twins_col(board)){
                                if(!twins_box(board)){
                                    return 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int main(int argc, char* argv[]){
    int num_threads = atoi(argv[1]);
    string input_file = argv[2];

    int* matrix = new int[N2];
    read_file(input_file, matrix);
    board_t* board = init_board(matrix);
    int index = 0;

    print_sudoku(board->matrix);
    cout<<endl;

    // for(int i = 0; i < N2; i++){
    //     if(board->matrix[i] != 0){
    //         cout<<board->matrix[i]<<" -- ";
    //         continue;
    //     }
            
    //     possible_values_in_cell(board, i);
    //     print_possible_values(board, i);
    //     cout<<" -- ";
    //     if((i+1)%N == 0){
    //         cout<<endl;
    //     }
    //     if((i+1)%N2 == 0){
    //         cout<<endl;
    //     }
    
    // }
    solve_partially(board);

    // for(int i = 0; i < N2; i++){
    //     if(board->matrix[i] != 0){
    //         cout<<board->matrix[i]<<" -- ";
    //         continue;
    //     }
            
    //     possible_values_in_cell(board, i);
    //     print_possible_values(board, i);
    //     cout<<" -- ";
    //     if((i+1)%N == 0){
    //         cout<<endl;
    //     }
    //     if((i+1)%N2 == 0){
    //         cout<<endl;
    //     }
    
    // }
    // print_all_possible_values(board);
    print_sudoku(board->matrix);

    // for(int i = 0; i < N2; i++){
    //     print_possible_values(board, i);
    //     cout<<"--";
    //     if((i+1)%N == 0){
    //         cout<<endl;
    //     }
    //     if((i+1)%N2 == 0){
    //         cout<<endl;
    //     }
    
    // }
    cout<<endl;
    return 0;
}