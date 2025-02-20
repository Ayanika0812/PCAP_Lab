#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

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
    int n;
    printf("Reg_No : 220905128\n");
    // Take user input for the size of the array
    printf("Enter the number of elements in the array: ");
    scanf("%d", &n);

    // Dynamically allocate memory for the array on the host
    int *h_arr = (int *)malloc(n * sizeof(int));

    // Seed the random number generator
    srand(time(NULL));

    // Generate random numbers for the array
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100;  // Random numbers between 0 and 99
    }

    // Print the original array
    printf("Original array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and 1 thread
    selectionSortKernel<<<1, 1>>>(d_arr, n);

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

 nvcc q2random.cu -o q2ran
student@lpcp-19:~/220905128/lab6$ ./q2ran
Enter the number of elements in the array: 20
Original array:
33 92 87 35 87 23 45 85 19 18 14 57 35 41 45 3 96 80 22 93 
Sorted array:
3 14 18 19 22 23 33 35 35 41 45 45 57 80 85 87 87 92 93 96 

*/