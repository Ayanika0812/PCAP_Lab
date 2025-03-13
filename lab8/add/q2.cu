#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper function to compute factorial of a number
__device__ int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Helper function to compute the sum of digits of a number
__device__ int sumOfDigits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// CUDA kernel to modify the matrix according to the problem's conditions
__global__ void modifyMatrix(int *A, int N) {
    int row = threadIdx.x;
    int col = threadIdx.y;

    // Principal diagonal elements set to zero
    if (row == col) {
        A[row * N + col] = 0;
    } 
    // Above diagonal elements replaced by their factorial
    else if (row < col) {
        A[row * N + col] = factorial(A[row * N + col]);
    } 
    // Below diagonal elements replaced by the sum of their digits
    else {
        A[row * N + col] = sumOfDigits(A[row * N + col]);
    }
}

// Function to print the matrix
void printMatrix(int *matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int N;

    // Take user input for matrix size
    printf("Enter the size of the matrix (N x N): ");
    scanf("%d", &N);

    // Allocate memory for matrix A on the host
    int *h_A = (int*)malloc(N * N * sizeof(int));

    // Initialize the matrix A with user input
    printf("Enter elements for matrix A:\n");
    for (int i = 0; i < N * N; i++) {
        scanf("%d", &h_A[i]);
    }

    // Allocate memory for matrix A on the device
    int *d_A;
    cudaMalloc((void**)&d_A, N * N * sizeof(int));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(N, N);
    dim3 gridDim(1, 1);

    // Launch the kernel to modify the matrix
    modifyMatrix<<<gridDim, blockDim>>>(d_A, N);

    // Copy the modified matrix back to the host
    cudaMemcpy(h_A, d_A, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the modified matrix
    printf("Modified Matrix A:\n");
    printMatrix(h_A, N);

    // Free memory
    free(h_A);
    cudaFree(d_A);

    return 0;
}

/*
./q2
Enter the size of the matrix (N x N): 3
Enter elements for matrix A:
4 4 3
5 3 6
1 3 2
Modified Matrix A:
0 24 6 
5 0 720 
1 3 0 
*/