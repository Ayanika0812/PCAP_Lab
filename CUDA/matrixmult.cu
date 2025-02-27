#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024  // Define the size of the matrix (N x N)

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void initializeMatrix(int *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = rand() % 10;  // Initialize with random values
    }
}

int main() {
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    // Allocate memory for host matrices
    h_A = (int *)malloc(N * N * sizeof(int));
    h_B = (int *)malloc(N * N * sizeof(int));
    h_C = (int *)malloc(N * N * sizeof(int));

    // Initialize matrices A and B with random values
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Allocate memory for device matrices
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    // Copy host matrices to device memory
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);  // Grid size to cover the entire matrix

    // Launch the kernel for matrix multiplication
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result matrix C back to host memory
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the resulting matrix (only the first 5x5 for brevity)
    printf("Resultant Matrix C (first 5x5 elements):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Matrix multiplication completed successfully!\n");

    return 0;
}
