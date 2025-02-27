#include <stdio.h>
#include <cuda_runtime.h>

__global__ void bubbleSortKernel(int *arr, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < n - 1; i++) {
        if (index < n - i - 1) {
            int temp = arr[index];
            if (arr[index] > arr[index + 1]) {
                arr[index] = arr[index + 1];
                arr[index + 1] = temp;
            }
        }
        __syncthreads();
    }
}

void bubbleSort(int *arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    bubbleSortKernel<<<numBlocks, blockSize>>>(d_arr, n);
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main() {
    int arr[] = {3, 2, 5, 1, 4};
    int n = sizeof(arr) / sizeof(arr[0]);

    bubbleSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    return 0;
}

/*
nvcc bubblesort.cu -o bubble
student@lpcp-19:~/220905128/CUDA$ ./bubble
Sorted array: 1 2 3 5 4 
*/