#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to compute the factorial in parallel using reduction
__global__ void factorial_kernel(long long *d_output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        d_output[idx] = idx + 1;  // Initialize each thread with its corresponding value
    } else {
        d_output[idx] = 1;  // Default value for threads that exceed 'n'
    }
    
    __syncthreads();  // Synchronize all threads before reduction

    // Perform parallel reduction (multiplying results together)
    for (int stride = 1; stride < n; stride *= 2) {
        int i = 2 * stride * idx;
        if (i < n) {
            d_output[i] *= d_output[i + stride];  // Reduce step: multiply two partial results
        }
        __syncthreads();  // Synchronize threads to ensure that reduction step completes
    }
}

void parallel_factorial(int n) {
    long long *d_output;
    long long h_output;

    // Allocate memory for output on device
    cudaMalloc((void**)&d_output, n * sizeof(long long));

    // Launch kernel to initialize factorial computation
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    factorial_kernel<<<blocks, threadsPerBlock>>>(d_output, n);

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(long long), cudaMemcpyDeviceToHost);

    printf("Factorial of %d is: %lld\n", n, h_output);

    // Free device memory
    cudaFree(d_output);
}

int main() {
    int n;
    printf("Enter a number to compute its factorial: ");
    scanf("%d", &n);

    parallel_factorial(n);

    return 0;
}
/*
nvcc fact.cu -o fact
student@lpcp-19:~/220905128/CUDA$ ./fact
Enter a number to compute its factorial: 4
Factorial of 4 is: 24
*/