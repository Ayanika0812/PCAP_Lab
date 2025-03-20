#include <stdio.h>
#include <cuda.h>

__global__ void addMatrixRowsAndCols(int *A, int *B, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        // Sum the ith row and jth column elements
        int rowSum = 0;
        int colSum = 0;

        // Sum the elements of the ith row
        for (int i = 0; i < N; i++) {
            rowSum += A[row * N + i];
        }

        // Sum the elements of the jth column
        for (int i = 0; i < M; i++) {
            colSum += A[i * N + col];
        }

        // Store the result in the output matrix
        B[row * N + col] = rowSum + colSum;
    }
}

void printMatrix(int *B, int M, int N) {
    printf("\nOutput Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i * N + j]);
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

    int h_A[M * N], h_B[M * N];
    printf("Enter matrix A elements in a single line (row-wise):\n");

    for (int i = 0; i < M * N; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A, *d_B;
    size_t size = M * N * sizeof(int);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Define the block and grid size
    dim3 blockDim(16, 16);  // block size of 16x16
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);  // grid size

    // Launch the kernel
    addMatrixRowsAndCols<<<gridDim, blockDim>>>(d_A, d_B, M, N);

    // Copy the result back to the host
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    // Print the output matrix
    printMatrix(h_B, M, N);

    return 0;
}


/*
nvcc q1.cu -o q1
student@lpcp-19:~/220905128/lab9/add$ ./q1
Enter the number of rows (M): 2
Enter the number of columns (N): 3
Enter matrix A elements in a single line (row-wise):
1 2 3
4 5 6

Output Matrix B:
11 13 15 
20 22 24 

*/