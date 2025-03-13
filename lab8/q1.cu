#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Function to print the matrix
void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Kernel to add matrices element-wise
__global__ void addKernel_elementwise(int *A, int *B, int *C, int rows, int cols) {
    int idx = threadIdx.y * cols + threadIdx.x;  // Calculate the index for the element in the matrix

    if (idx < rows * cols) {
        C[idx] = A[idx] + B[idx];  // Perform element-wise addition
    }
}

// Kernel to add matrices column-wise
__global__ void addKernel_colwise(int *A, int *B, int *C, int rows, int cols) {
    int cidB = threadIdx.x;  // Column index in B (and C)
    //int sum;

    for (int ridA = 0; ridA < rows; ridA++) {
        C[ridA * cols + cidB] = A[ridA * cols + cidB] + B[ridA * cols + cidB];  // Column-wise addition
    }
}

// Kernel to add matrices row-wise
__global__ void addKernel_rowwise(int *A, int *B, int *C, int cols) {
    int ridA = threadIdx.x;  // Row index in A (and C)
    
    for (int cidB = 0; cidB < cols; cidB++) {
        C[ridA * cols + cidB] = A[ridA * cols + cidB] + B[ridA * cols + cidB];  // Row-wise addition
    }
}

int main() {
    int rows, cols;

    // Take user input for matrix dimensions
    printf("Enter the number of rows for matrix A (and B): ");
    scanf("%d", &rows);
    printf("Enter the number of columns for matrix A (and B): ");
    scanf("%d", &cols);

    // Allocate memory for matrices on the host
    int *h_A = (int*)malloc(rows * cols * sizeof(int));
    int *h_B = (int*)malloc(rows * cols * sizeof(int));
    int *h_C = (int*)malloc(rows * cols * sizeof(int));

    // Initialize matrices A and B with user input
    printf("Enter elements for matrix A:\n");
    for (int i = 0; i < rows * cols; i++) {
        scanf("%d", &h_A[i]);
    }

    printf("Enter elements for matrix B:\n");
    for (int i = 0; i < rows * cols; i++) {
        scanf("%d", &h_B[i]);
    }

    int *d_A, *d_B, *d_C;

    // Allocate memory for matrices on the device
    cudaMalloc((void**)&d_A, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    // Add matrices element-wise (one thread per element)
    dim3 blockDim_elementwise(cols, rows);  // Enough threads for each element
    dim3 gridDim_elementwise(1, 1);         // One block
    addKernel_elementwise<<<gridDim_elementwise, blockDim_elementwise>>>(d_A, d_B, d_C, rows, cols);
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix Addition (Element-wise):\n");
    printMatrix(h_C, rows, cols);

    // Reset the result matrix
    cudaMemset(d_C, 0, rows * cols * sizeof(int));

    // Add matrices column-wise (one thread per column)
    dim3 blockDim_colwise(cols, 1);  // One thread per column
    dim3 gridDim_colwise(1, 1);      // One block
    addKernel_colwise<<<gridDim_colwise, blockDim_colwise>>>(d_A, d_B, d_C, rows, cols);
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix Addition (Column-wise):\n");
    printMatrix(h_C, rows, cols);

    // Reset the result matrix
    cudaMemset(d_C, 0, rows * cols * sizeof(int));

    // Add matrices row-wise (one thread per row)
    dim3 blockDim_rowwise(rows, 1);   // One thread per row
    dim3 gridDim_rowwise(1, 1);       // One block
    addKernel_rowwise<<<gridDim_rowwise, blockDim_rowwise>>>(d_A, d_B, d_C, cols);
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix Addition (Row-wise):\n");
    printMatrix(h_C, rows, cols);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
/*
nvcc q1.cu -o q1
student@lpcp-19:~/220905128/lab8$ ./q1
Enter the number of rows for matrix A (and B): 3
Enter the number of columns for matrix A (and B): 3
Enter elements for matrix A:
1 2 3
1 1 1
1 2 4
Enter elements for matrix B:
1 1 1
1 1 1
1 1 1
Matrix Addition (Element-wise):
2 3 4 
2 2 2 
2 3 5 
Matrix Addition (Column-wise):
2 3 4 
2 2 2 
2 3 5 
Matrix Addition (Row-wise):
2 3 4 
2 2 2 
2 3 5 
*/