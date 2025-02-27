#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to perform parallel linear search
__global__ void parallel_linear_search(int *d_arr, int n, int target, bool *d_found) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        if (d_arr[idx] == target) {
            *d_found = true;  // Found the target, update the found flag
        }
    }
}

int main() {
    int n, target;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int h_arr[n];
    printf("Enter the elements of the array:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_arr[i]);
    }

    printf("Enter the target value to search: ");
    scanf("%d", &target);

    // Allocate memory on the device
    int *d_arr;
    bool *d_found;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_found, sizeof(bool));

    // Copy array to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_found, false, sizeof(bool));  // Initialize found flag to false

    // Launch the kernel to perform parallel linear search
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    parallel_linear_search<<<blocks, threadsPerBlock>>>(d_arr, n, target, d_found);

    // Copy result back to host
    bool h_found;
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    if (h_found) {
        printf("Target %d found in the array.\n", target);
    } else {
        printf("Target %d not found in the array.\n", target);
    }

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_found);

    return 0;
}
/*
student@lpcp-19:~/220905128/CUDA$ nvcc linseach.cu -o linear
student@lpcp-19:~/220905128/CUDA$ ./linear
Enter the size of the array: 5
Enter the elements of the array:
3 6 9 7 0
Enter the target value to search: 9
Target 9 found in the array.
*/