#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0;
    for (int i = 0; i < ceil((float)A_cols / TILE_WIDTH); i++) {
        __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
        __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
        
        if (row < A_rows && (i * TILE_WIDTH + threadIdx.x) < A_cols) {
            tileA[threadIdx.y][threadIdx.x] = A[row * A_cols + (i * TILE_WIDTH + threadIdx.x)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < B_cols && (i * TILE_WIDTH + threadIdx.y) < A_cols) {
            tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * B_cols + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();  
        for (int j = 0; j < TILE_WIDTH; j++) {
            value += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        } 
        __syncthreads();
    }
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
    }
}
void matrixMultiply(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, A_rows * A_cols * sizeof(float));
    cudaMalloc((void**)&d_B, A_cols * B_cols * sizeof(float));
    cudaMalloc((void**)&d_C, A_rows * B_cols * sizeof(float));
    cudaMemcpy(d_A, A, A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, A_cols * B_cols * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks(ceil((float)B_cols / TILE_WIDTH), ceil((float)A_rows / TILE_WIDTH));
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    cudaMemcpy(C, d_C, A_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
int main() {
    int A_rows, A_cols, B_rows, B_cols;
    printf("Enter the number of rows and columns for matrix A: ");
    scanf("%d %d", &A_rows, &A_cols);
    printf("Enter the number of rows and columns for matrix B: ");
    scanf("%d %d", &B_rows, &B_cols);
    if (A_cols != B_rows) {
        printf("Matrix multiplication is not possible\n");
        return -1;
    }
    float *A = (float*)malloc(A_rows * A_cols * sizeof(float));
    float *B = (float*)malloc(B_rows * B_cols * sizeof(float));
    float *C = (float*)malloc(A_rows * B_cols * sizeof(float));
    printf("Enter elements for matrix A row-wise (%d elements):\n", A_rows * A_cols);
    for (int i = 0; i < A_rows * A_cols; i++) {
        scanf("%f", &A[i]);
    }
    printf("Enter elements for matrix B row-wise (%d elements):\n", B_rows * B_cols);
    for (int i = 0; i < B_rows * B_cols; i++) {
        scanf("%f", &B[i]);
    }
    matrixMultiply(A, B, C, A_rows, A_cols, B_cols);
    printf("Result Matrix C:\n");
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            printf("%.2f ", C[i * B_cols + j]);
        }
        printf("\n");
    }
    free(A);
    free(B);
    free(C);
    return 0;
}


/*
./q1
Enter the number of rows and columns for matrix A: 3 2
Enter the number of rows and columns for matrix B: 2 4
Enter elements for matrix A row-wise (6 elements):
1 2
1 4
1
^C
student@lpcp-19:~/220905128/lab10$ ./q1
Enter the number of rows and columns for matrix A: 3 3
Enter the number of rows and columns for matrix B: 3 3
Enter elements for matrix A row-wise (9 elements):
1 3 4
2 5 6
7 8 9
Enter elements for matrix B row-wise (9 elements):
3 8 5  
8 2 1
9 0 2
Result Matrix C:
63.00 14.00 16.00 
100.00 26.00 27.00 
166.00 72.00 61.00 
*/