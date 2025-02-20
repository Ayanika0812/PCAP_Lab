#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// CUDA kernel for the even phase
__global__ void evenPhaseKernel(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Compare and swap elements at even indices
    if (i < n / 2 && (2 * i) < n - 1) {
        if (arr[2 * i] > arr[2 * i + 1]) {
            int temp = arr[2 * i];
            arr[2 * i] = arr[2 * i + 1];
            arr[2 * i + 1] = temp;
        }
    }
}

// CUDA kernel for the odd phase
__global__ void oddPhaseKernel(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Compare and swap elements at odd indices
    if (i < n / 2 && (2 * i + 1) < n - 1) {
        if (arr[2 * i + 1] > arr[2 * i + 2]) {
            int temp = arr[2 * i + 1];
            arr[2 * i + 1] = arr[2 * i + 2];
            arr[2 * i + 2] = temp;
        }
    }
}

// Host code
int main() {
    int n;
    printf("Reg_No : 220905128\n");
    printf("Enter the number of elements in the array: ");
    scanf("%d", &n);

    // Dynamically allocate memory for the host array
    int *h_arr = (int *)malloc(n * sizeof(int));

    // Seed the random number generator
    srand(time(NULL));

    // Generate random numbers for the array
    printf("Original array:\n");
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100;  // Random numbers between 0 and 99
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_arr;
    cudaMalloc((void **)&d_arr, n * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Perform odd-even transposition sort
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            // Even phase
            evenPhaseKernel<<<gridSize, blockSize>>>(d_arr, n);
        } else {
            // Odd phase
            oddPhaseKernel<<<gridSize, blockSize>>>(d_arr, n);
        }
        cudaDeviceSynchronize();  // Ensure all threads are synchronized
    }

    // Copy the sorted array back to the host
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory and host memory
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}


/*

nvcc q3.cu -o q3
student@lpcp-19:~/220905128/lab6$ ./q3
Enter the number of elements in the array: 10
Original array:
64 48 9 94 56 74 67 0 55 37 
Sorted array:
0 9 37 48 55 56 64 67 74 94 

*/