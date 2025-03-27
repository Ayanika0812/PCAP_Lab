
#include <stdio.h>
#include <cuda.h>

#define MAX_N 1024
#define MAX_FILTER_WIDTH 32

__constant__ float d_filter[MAX_FILTER_WIDTH];

__global__ void convolution_1d(float *input, float *output, int n, int filter_width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = filter_width / 2;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < filter_width; i++) {
            int input_idx = tid - r + i;
            if (input_idx >= 0 && input_idx < n) {
                sum += input[input_idx] * d_filter[i];
            }
        }
        output[tid] = sum;
    }
}

int main() {
    int n, filter_width;

    printf("Enter size of input signal (<= %d): ", MAX_N);
    scanf("%d", &n);
    if (n > MAX_N) return -1;

    printf("Enter filter width (odd number <= %d): ", MAX_FILTER_WIDTH);
    scanf("%d", &filter_width);
    if (filter_width > MAX_FILTER_WIDTH || filter_width % 2 == 0) return -1;

    float h_input[MAX_N], h_output[MAX_N], h_filter[MAX_FILTER_WIDTH];

    printf("Enter %d input signal values:\n", n);
    for (int i = 0; i < n; i++) scanf("%f", &h_input[i]);

    printf("Enter %d filter values:\n", filter_width);
    for (int i = 0; i < filter_width; i++) scanf("%f", &h_filter[i]);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, h_filter, filter_width * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    convolution_1d<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, filter_width);

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Convolved Output:\n");
    for (int i = 0; i < n; i++) printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

/*
 ./q2
Enter size of input signal (<= 1024): 6
Enter filter width (odd number <= 32): 3
Enter 6 input signal values:
3 5 6 1 8 2
Enter 3 filter values:
3 5 7
Convolved Output:
50.000000 76.000000 52.000000 79.000000 57.000000 34.000000 
*/