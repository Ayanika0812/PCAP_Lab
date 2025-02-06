#include<stdio.h>
#include<stdlib.h>

__global__ void vecAddKernel(float *A,float *B,float *C,int n)
{
    //int i = threadIdx.x + blockDim.x * blockIdx.x // 1D grid
    //int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y 
	int i = threadIdx.x;

	printf("")
}