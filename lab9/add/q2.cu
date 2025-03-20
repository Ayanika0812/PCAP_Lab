#include <stdio.h>
#include <cuda_runtime.h>

__device__ void append_char_to_output(char* str, int index, char c, int repeat_count) {
    for (int i = 0; i < repeat_count; i++) {
        str[index + i] = c;
    }
}

__global__ void repeat_characters_kernel(char* A, int* B, char* output, int m, int n, int* atomic_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m * n) {
        int row = idx / n;
        int col = idx % n;

        char char_a = A[idx];
        int repeat_count = B[idx];

        // Get the current position to insert the characters using atomic counter
        int output_index = atomicAdd(atomic_counter, repeat_count);

        append_char_to_output(output, output_index, char_a, repeat_count);
    }
}

int main() {
    int M, N;

    // Taking input for matrix dimensions
    printf("Enter the number of rows (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns (N): ");
    scanf("%d", &N);

    char *A = (char*)malloc(M * N * sizeof(char));
    int *B = (int*)malloc(M * N * sizeof(int));

    // Taking input for character matrix A
    printf("Enter the character matrix A (%d x %d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf(" %c", &A[i * N + j]); // " %c" ensures it reads the character correctly
        }
    }

    // Taking input for integer matrix B
    printf("Enter the integer matrix B (%d x %d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &B[i * N + j]);
        }
    }

    char *d_A, *d_output;
    int *d_B;
    int atomic_counter = 0;

    int output_size = M * N * 5;  // Assuming max repetitions of 5

    // Allocating memory on the device
    cudaMalloc(&d_A, M * N * sizeof(char));
    cudaMalloc(&d_B, M * N * sizeof(int));
    cudaMalloc(&d_output, output_size * sizeof(char));
    int *d_atomic_counter;
    cudaMalloc(&d_atomic_counter, sizeof(int));  // Allocate memory for atomic counter

    // Copying data from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_atomic_counter, &atomic_counter, sizeof(int), cudaMemcpyHostToDevice);

    // Initialize output string with null characters
    char init_output[output_size] = {0};
    cudaMemcpy(d_output, init_output, output_size * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel with appropriate grid and block sizes
    int blockSize = 256;
    int gridSize = (M * N + blockSize - 1) / blockSize;

    repeat_characters_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_output, M, N, d_atomic_counter);

    // Check for any errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy result back to host
    char h_output[output_size];
    cudaMemcpy(h_output, d_output, output_size * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the output string, skipping any unused characters (0s)
    printf("Output String: ");
    for (int i = 0; i < output_size; i++) {
        if (h_output[i] != 0) {
            printf("%c", h_output[i]);
        }
    }
    printf("\n");

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);
    cudaFree(d_atomic_counter);

    free(A);
    free(B);

    return 0;
}
