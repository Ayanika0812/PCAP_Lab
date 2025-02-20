#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

// Define mask width
#define MASK_WIDTH 5

// CUDA kernel for 1D convolution
__global__ void convolution1D(float *N, float *M, float *P, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform convolution only if within bounds
    if (i < width) {
        float result = 0.0;
        
        // Apply the mask
        for (int j = 0; j < MASK_WIDTH; j++) {
            int index = i - (MASK_WIDTH / 2) + j;
            if (index >= 0 && index < width) { // Check for boundary conditions
                result += N[index] * M[j];
            }
        }

        // Store the result
        P[i] = result;
    }
}

int main() {
    int width;
    printf("Reg_No : 220905128\n");
    // Take user input for array size
    printf("Enter the number of elements in the array: ");
    scanf("%d", &width);

    // Seed the random number generator
    srand(time(NULL));

    // Dynamically allocate input and output arrays
    float *h_N = (float *)malloc(width * sizeof(float));
    float h_M[MASK_WIDTH] = {2.0, 2.0, 5.0, 2.0, 1.0};  // Fixed mask array
    float *h_P = (float *)malloc(width * sizeof(float));

    // Generate random numbers for the input array
    printf("Generated input array:\n");
    for (int i = 0; i < width; i++) {
        h_N[i] = (float)(rand() % 10);  // Random numbers between 0 and 9
        printf("%f ", h_N[i]);
    }
    printf("\n");

    // Allocate device memory
    float *d_N, *d_M, *d_P;
    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_M, MASK_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));

    // Copy input and mask to device
    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;

    // Launch the kernel
    convolution1D<<<gridSize, blockSize>>>(d_N, d_M, d_P, width);

    // Copy the result back to the host
    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Convolution result:\n");
    for (int i = 0; i < width; i++) {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    // Free host memory
    free(h_N);
    free(h_P);

    return 0;
}





/*
nvcc q1.cu -o q1
student@lpcp-19:~/220905128/lab6$ ./q1
Enter the number of elements in the array: 7
Generated input array:
9.0 2.0 8.0 2.0 9.0 2.0 8.0 
Mask array:
2.0 2.0 5.0 2.0 1.0 
Convolution result:
57.0 46.0 75.0 50.0 77.0 48.0 62.0 



P[0] = (9.0 * 5.0) + (2.0 * 2.0) + (8.0 * 1.0)
P[0] = 45.0 + 4.0 + 8.0 = 57.0

*/