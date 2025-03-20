#include <stdio.h>
#include <cuda.h>

__global__ void generateOutputString(char *A, int *B, char *output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the index in the matrix A and B
    if (row < M && col < N) {
        int index = row * N + col;
        char character = A[index];
        int repeatCount = B[index];

        // Calculate the position to write the output (index for the thread)
        int startIdx = 0;
        for (int i = 0; i < index; i++) {
            startIdx += B[i];  // Sum the previous counts to get the start index
        }

        // Write the repeated characters into the output array
        for (int i = 0; i < repeatCount; i++) {
            output[startIdx + i] = character;
        }
    }
}

void printString(char *output, int totalLength) {
    for (int i = 0; i < totalLength; i++) {
        printf("%c", output[i]);
    }
    printf("\n");
}

int main() {
    int M, N;
    printf("Enter the number of rows (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns (N): ");
    scanf("%d", &N);

    // Allocate memory for input matrices
    char h_A[M * N];
    int h_B[M * N];

    // Input matrix A (characters)
    printf("Enter matrix A (characters):\n");
    for (int i = 0; i < M * N; i++) {
        scanf(" %c", &h_A[i]);  // Note the space before %c to capture single characters correctly
    }

    // Input matrix B (repetition counts)
    printf("Enter matrix B (integer repetition counts):\n");
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &h_B[i]);
    }

    // Calculate the total number of characters in the output string
    int totalLength = 0;
    for (int i = 0; i < M * N; i++) {
        totalLength += h_B[i];  // Sum of all repetition counts
    }

    // Allocate memory for output string
    char *h_output = (char *)malloc(totalLength * sizeof(char));
    char *d_A, *d_B, *d_output;

    // Allocate device memory
    cudaMalloc((void **)&d_A, M * N * sizeof(char));
    cudaMalloc((void **)&d_B, M * N * sizeof(int));
    cudaMalloc((void **)&d_output, totalLength * sizeof(char));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    generateOutputString<<<gridDim, blockDim>>>(d_A, d_B, d_output, M, N);

    // Copy the output string back to host
    cudaMemcpy(h_output, d_output, totalLength * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Output String STR: ");
    printString(h_output, totalLength);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);

    // Free host memory
    free(h_output);

    return 0;
}
