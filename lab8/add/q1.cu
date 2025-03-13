#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void computeRowSumAndColSum(int *A, int *B, int M, int N) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    
    // Shared memory to hold row sums and column sums
    __shared__ int rowSum[1024]; // Assuming M < 1024
    __shared__ int colSum[1024]; // Assuming N < 1024
    
    if (col == 0) {
        // Compute row sum for this row
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j];
        }
        rowSum[row] = sum;
    }

    if (row == 0) {
        // Compute column sum for this column
        int sum = 0;
        for (int i = 0; i < M; i++) {
            sum += A[i * N + col];
        }
        colSum[col] = sum;
    }

    __syncthreads();

    // Now set the value of matrix B
    if (A[row * N + col] % 2 == 0) { // Even number -> row sum
        B[row * N + col] = rowSum[row];
    } else { // Odd number -> column sum
        B[row * N + col] = colSum[col];
    }
}

void printMatrix(int *matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int M, N;

    printf("Enter the number of rows (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns (N): ");
    scanf("%d", &N);

    int *h_A = (int*)malloc(M * N * sizeof(int));
    int *h_B = (int*)malloc(M * N * sizeof(int));

    printf("Enter elements for matrix A:\n");
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A, *d_B;
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // One block of threads, each thread computes one element in matrix B
    dim3 blockDim(M, N);
    dim3 gridDim(1, 1); // One block

    computeRowSumAndColSum<<<gridDim, blockDim>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant Matrix B:\n");
    printMatrix(h_B, M, N);

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}


/*
nvcc q1.cu -o q1
^[[Astudent@lpcp-19:~/220905128/lab8/add$ ./q1
Enter the number of rows (M): 2
Enter the number of columns (N): 3
Enter elements for matrix A:
1 2 3
4 5 6
Resultant Matrix B:
5 6 9 
15 7 15 
*/