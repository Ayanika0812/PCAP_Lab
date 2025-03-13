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

// Kernel to multiply matrices element-wise
__global__ void multiplyKernel_elementwise(int *A, int *B, int *C, int wa) {
    int ridA = threadIdx.y;  // Row index in A
    int cidB = threadIdx.x;  // Column index in B
    int wb = blockDim.x;
    int sum = 0, k;

    for (k = 0; k < wa; k++) {
        sum += (A[ridA * wa + k] * B[k * wb + cidB]);
    }
    C[ridA * wb + cidB] = sum;
}

// Kernel to multiply matrices column-wise
__global__ void multiplyKernel_colwise(int *A, int *B, int *C, int ha, int wa) {
    int cidB = threadIdx.x;  // Column index in B
    int wb = blockDim.x;
    int sum, k;

    for (int ridA = 0; ridA < ha; ridA++) {
        sum = 0;
        for (k = 0; k < wa; k++) {
            sum += (A[ridA * wa + k] * B[k * wb + cidB]);
        }
        C[ridA * wb + cidB] = sum;
    }
}

// Kernel to multiply matrices row-wise
__global__ void multiplyKernel_rowwise(int *A, int *B, int *C, int wa, int wb) {
    int ridA = threadIdx.x;  // Row index in A
    int sum;

    for (int cidB = 0; cidB < wb; cidB++) {
        sum = 0;
        for (int k = 0; k < wa; k++) {
            sum += (A[ridA * wa + k] * B[k * wb + cidB]);
        }
        C[ridA * wb + cidB] = sum;
    }
}

int main() {
    int ha, wa, wb;

    // Take user input for matrix dimensions
    printf("Enter number of rows for matrix A (ha): ");
    scanf("%d", &ha);
    printf("Enter number of columns for matrix A / rows for matrix B (wa): ");
    scanf("%d", &wa);
    printf("Enter number of columns for matrix B (wb): ");
    scanf("%d", &wb);

    // Allocate memory for matrices on the host
    int *h_A = (int*)malloc(ha * wa * sizeof(int));
    int *h_B = (int*)malloc(wa * wb * sizeof(int));
    int *h_C = (int*)malloc(ha * wb * sizeof(int));

    // Initialize matrices A and B with user input
    printf("Enter elements for matrix A:\n");
    for (int i = 0; i < ha * wa; i++) {
        scanf("%d", &h_A[i]);
    }

    printf("Enter elements for matrix B:\n");
    for (int i = 0; i < wa * wb; i++) {
        scanf("%d", &h_B[i]);
    }

    int *d_A, *d_B, *d_C;

    // Allocate memory for matrices on the device
    cudaMalloc((void**)&d_A, ha * wa * sizeof(int));
    cudaMalloc((void**)&d_B, wa * wb * sizeof(int));
    cudaMalloc((void**)&d_C, ha * wb * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, ha * wa * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, wa * wb * sizeof(int), cudaMemcpyHostToDevice);

    // Multiply matrices element-wise (one thread per element)
    dim3 blockDim_elementwise(wb, ha);  // Enough threads for each element
    dim3 gridDim_elementwise(1, 1);     // One block
    multiplyKernel_elementwise<<<gridDim_elementwise, blockDim_elementwise>>>(d_A, d_B, d_C, wa);
    cudaMemcpy(h_C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix Multiplication (Element-wise):\n");
    printMatrix(h_C, ha, wb);

    // Reset the result matrix
    cudaMemset(d_C, 0, ha * wb * sizeof(int));

    // Multiply matrices column-wise (one thread per column)
    dim3 blockDim_colwise(wb, 1);  // One thread per column
    dim3 gridDim_colwise(1, 1);    // One block
    multiplyKernel_colwise<<<gridDim_colwise, blockDim_colwise>>>(d_A, d_B, d_C, ha, wa);
    cudaMemcpy(h_C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix Multiplication (Column-wise):\n");
    printMatrix(h_C, ha, wb);

    // Reset the result matrix
    cudaMemset(d_C, 0, ha * wb * sizeof(int));

    // Multiply matrices row-wise (one thread per row)
    dim3 blockDim_rowwise(ha, 1);   // One thread per row
    dim3 gridDim_rowwise(1, 1);     // One block
    multiplyKernel_rowwise<<<gridDim_rowwise, blockDim_rowwise>>>(d_A, d_B, d_C, wa, wb);
    cudaMemcpy(h_C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix Multiplication (Row-wise):\n");
    printMatrix(h_C, ha, wb);

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
nvcc q2.cu -o q2
student@lpcp-19:~/220905128/lab8$ ./q2
Enter number of rows for matrix A (ha): 3
Enter number of columns for matrix A / rows for matrix B (wa): 3
Enter number of columns for matrix B (wb): 3
Enter elements for matrix A:
1 2 3
1 1 1
1 2 4
Enter elements for matrix B:
1 1 1
1 1 1
1 1 1
Matrix Multiplication (Element-wise):
6 6 6 
3 3 3 
7 7 7 
Matrix Multiplication (Column-wise):
6 6 6 
3 3 3 
7 7 7 
Matrix Multiplication (Row-wise):
6 6 6 
3 3 3 
7 7 7 
*/