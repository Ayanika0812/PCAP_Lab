#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void scanKernel(int *d_input, int *d_output, int N) {
    extern __shared__ int shared_data[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    if (idx < N) {
        shared_data[thread_id] = d_input[idx];
    } else {
        shared_data[thread_id] = 0;
    }
    __syncthreads();
    // Inclusive scan (prefix sum)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (thread_id >= stride) {
            temp = shared_data[thread_id - stride];
        }
        __syncthreads();
        shared_data[thread_id] += temp;
        __syncthreads();
    }
    if (idx < N) {
        d_output[idx] = shared_data[thread_id];
    }
}

void scan(int *h_input, int *h_output, int N) {
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    scanKernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int N;
    printf("Enter the number of elements: ");
    scanf("%d", &N);
    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(N * sizeof(int));
    printf("Enter the elements of the input array:\n");
    for (int i = 0; i < N; i++) {
        scanf("%d", &h_input[i]);
    }
    scan(h_input, h_output, N);
    printf("Input array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");
    printf("Inclusive scan result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    free(h_input);
    free(h_output);
    return 0;
}


/
*
nvcc l10q3.cu -o q3
student@lpcp-19:~/220905128/week11$ ./q3
Enter the number of elements: 4
Enter the elements of the input array:
10 20 30 40
Input array:
10 20 30 40 
Inclusive scan result:
10 30 60 100 
*/