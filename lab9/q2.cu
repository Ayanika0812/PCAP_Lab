#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to process each element in the matrix
__global__ void processMatrix(int *A, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are within the bounds of the matrix
    if (row < M && col < N) {
        int index = row * N + col;
        int power = row + 1; // The row number determines the power (1 for first row, 2 for second row, etc.)
        
        // Raise the element to the power of (row+1)
        int value = A[index];
        int result = 1;
        for (int i = 0; i < power; i++) {
            result *= value;
        }
        
        A[index] = result;  // Update the element in the matrix
    }
}

int main()
{
    int M, N;
    
    // Input dimensions of the matrix
    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    int *A = (int *)malloc(M * N * sizeof(int));  // Host matrix A

    // Input matrix elements
    printf("Enter the elements of the matrix A (%d x %d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &A[i * N + j]);
        }
    }

    // Allocate memory for matrix on the device
    int *d_A;
    cudaMalloc((void **)&d_A, M * N * sizeof(int));

    // Copy matrix from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    processMatrix<<<gridSize, blockSize>>>(d_A, M, N);

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Copy the result matrix back to host
    cudaMemcpy(A, d_A, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the modified matrix
    printf("Modified matrix A after transformation:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\nReg No. - 220905128\n");
    // Free allocated memory
    cudaFree(d_A);
    free(A);
    
    return 0;
}


/*
nvcc q2.cu -o q2
student@lpcp-19:~/220905128/lab9$ ./q2
Enter number of rows (M): 3
Enter number of columns (N): 3
Enter the elements of the matrix A (3 x 3):
1 4 3
2 3 4
3 2 1
Modified matrix A after transformation:
1 4 3 
4 9 16 
27 8 1 

*/