#include<stdio.h>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

__device__ int getGTID(){
    int blockid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadid = blockid * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadid;
}

__global__ void add(int *a, int *b, int *c, int *n){
    int gtid = getGTID();
    if (gtid < *n){
        c[gtid] = a[gtid] + b[gtid];
    }
}

int main(){
    int *a, *b, *c;
    int n;   
    printf("Enter the size of the vectors: ");
    scanf("%d", &n);

    int s = n * sizeof(int);

    a = (int *)malloc(s);
    b = (int *)malloc(s);
    c = (int *)malloc(s);

    // Hardcode values for arrays a and b
    for (int i = 0; i < n; i++) {
        a[i] = i + 1;  // Array A: 1, 2, 3, ..., n
        b[i] = (i + 1) * 2;  // Array B: 2, 4, 6, ..., 2n
    }

    int *d_a, *d_b, *d_c, *d_n;
    cudaMalloc((void **)&d_a, s);
    cudaMalloc((void **)&d_b, s);
    cudaMalloc((void **)&d_c, s);
    cudaMalloc((void **)&d_n, sizeof(int));  // treat it as a pointer only
    
    cudaMemcpy(d_a, a, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    add<<<ceil(n / 256.0), 256>>>(d_a, d_b, d_c, d_n);  // needs to be float for ceil
    cudaMemcpy(c, d_c, s, cudaMemcpyDeviceToHost);

    // Display the unused threads for efficiency
    printf("Unused : %.0f   \n", (ceil(n / 256.0) * 256) - n);

    // Display the arrays a, b, and c
    printf("Array A: ");
    for (int i = 0; i < n; i++) {
        printf("%d  ", a[i]);
    }
    printf("\n");

    printf("Array B: ");
    for (int i = 0; i < n; i++) {
        printf("%d  ", b[i]);
    }
    printf("\n");

    // Display the result of C array
    printf("Result C (A + B): ");
    for (int i = 0; i < n; i++) {
        printf("%d  ", c[i]);
    }
    printf("\n");

    // Free the GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Reg no. 220905128\n");  
}
