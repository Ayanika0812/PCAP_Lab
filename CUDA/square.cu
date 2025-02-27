#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to square each element of the array
__global__ void squareArray(int *d_arr, int *d_result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index
    if (idx < n) {
        d_result[idx] = d_arr[idx] * d_arr[idx];  // Square the element
    }
}

int main() {
    int n;

    // Take input from the user
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int *h_arr = (int*)malloc(n * sizeof(int));  // Host array
    int *h_result = (int*)malloc(n * sizeof(int));  // Array to store results on the host

    // Take array elements from the user
    printf("Enter the elements of the array: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_arr[i]);
    }

    int *d_arr, *d_result;  // Pointers for device memory

    // Allocate device memory
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_result, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks needed

    // Launch the kernel
    squareArray<<<numBlocks, blockSize>>>(d_arr, d_result, n);

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Squared Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_result[i]);
    }
    printf("\n");

    // Free device memory and host memory
    cudaFree(d_arr);
    cudaFree(d_result);
    free(h_arr);
    free(h_result);

    return 0;
}


/*

nvcc square.cu -o sq
student@lpcp-19:~/220905128/CUDA$ ./sq
Enter the number of elements: 4
Enter the elements of the array: 1 2 3 4
Squared Array: 1 4 9 16 

*/