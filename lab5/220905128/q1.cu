#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add_a(int *a, int *b, int *c){
    int i = blockIdx.x;
    c[i] = a[i] + b[i];  
}

__global__ void add_b(int *a, int *b, int *c){
    int i = threadIdx.x;
    c[i] = a[i] + b[i]; 
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

    printf("Enter values of array A: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }
    printf("Enter values of array B: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &b[i]);
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, s);
    cudaMalloc((void **)&d_b, s);
    cudaMalloc((void **)&d_c, s);

    cudaMemcpy(d_a, a, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, s, cudaMemcpyHostToDevice);

    // Method b: 1 block, N threads
    add_b<<<1, n>>>(d_a, d_b, d_c);   
    cudaMemcpy(c, d_c, s, cudaMemcpyDeviceToHost);
    printf("Result C by method b: <<<1, n>>>:  ");
    for (int i = 0; i < n; i++) {
        printf("%d  ", c[i]);
    }
    printf("\n");
    cudaFree(d_c);
    // Method a: N blocks, 1 thread per block
    add_a<<<n, 1>>>(d_a, d_b, d_c);
    cudaMemcpy(c, d_c, s, cudaMemcpyDeviceToHost);
    printf("Result C by method a: <<<n, 1>>>  : ");
    for (int i = 0; i < n; i++) {
        printf("%d  ", c[i]);
    }
    printf("\n");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    printf("Reg no. 220905128\n");
    return 0;

}


/*
Astudent@lpcp-19:~/220905128/lab5$ nvcc q1.cu -o q1
student@lpcp-19:~/220905128/lab5$ ./q1
Enter the size of the vectors: 5
Enter values of array A: 1 2 3 4 5
Enter values of array B: 6 7 8 9 10
Result C by method b: <<<1, n>>>:  7  9  11  13  15  
Result C by method a: <<<n, 1>>>  : 7  9  11  13  15  

*/