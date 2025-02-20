#include <stdio.h>
#include <cuda.h>

// CUDA kernel to perform selection sort
__global__ void selectionSortKernel(int *arr, int n) {
    int i, j, minIdx;
    for (i = 0; i < n - 1; i++) {
        minIdx = i;

        // Find minimum element in the unsorted part of the array
        for (j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }

        // Swap the minimum element with the first element of the unsorted part
        if (minIdx != i) {
            int temp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = temp;
        }

        // Ensure all threads are synchronized before moving to the next iteration
        __syncthreads();
    }
}

// Host code
int main() {
	printf("Reg_No : 220905128\n");
    const int n = 8;
    int h_arr[n] = {64, 34, 25, 12, 22, 11, 90, 33};  // Example array

    // Allocate device memory
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and 1 thread (selection sort is inherently sequential)
    selectionSortKernel<<<1, 1>>>(d_arr, n);

    // Copy the sorted array back to the host
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);

    return 0;
}
