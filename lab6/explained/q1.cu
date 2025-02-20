#include <stdio.h>
#include <cuda.h>

// Define mask width
#define MASK_WIDTH 5

// CUDA kernel for 1D convolution
__global__ void convolution1D(float *N, float *M, float *P, int width) {
    // Calculate the thread's global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform convolution only if within bounds
    if (i < width) {
        float result = 0.0;
        
        // Apply the mask
        for (int j = 0; j < MASK_WIDTH; j++) {
            int index = i - (MASK_WIDTH / 2) + j;
            if (index >= 0 && index < width) { // Check for boundary conditions
                result += N[index] * M[j];
            }
        }

        // Store the result
        P[i] = result;
    }
}

// Host code
int main() {
    const int width = 7; // Size of input array
    float h_N[width] = {1,2,3,4,5,6,7};  // Example input array
    float h_M[MASK_WIDTH] = {3,4,5,4,3};  // Example mask

    // Allocate memory for output array
    float h_P[width];

    // Allocate device memory
    float *d_N, *d_M, *d_P;
    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_M, MASK_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));

    // Copy input and mask to device
    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;

    // Launch the kernel
    convolution1D<<<gridSize, blockSize>>>(d_N, d_M, d_P, width);

    // Copy the result back to the host
    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Convolution result:\n");
    for (int i = 0; i < width; i++) {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}


/*
 nvcc q1.cu -o q1
 ./q1
Convolution result:
22.000000 38.000000 57.000000 76.000000 95.000000 90.000000 74.000000 

*/


/*

For P[i]P[i], the general formula is:
P[i]=∑j=0MASK_WIDTH−1 N[i−offset+j]×M[j]
P[i]=j=0∑MASK_WIDTH−1 N[i−offset+j]×M[j]    

Where:

    Nis the input array.
    M is the mask.
    offset=MASK_WIDTH/2
    offset=2

For MASK_WIDTH=5

Calculation for P[0]P[0]:

    Mask Centering: At index 0, we center the mask over the element N[0]N[0], so the mask’s midpoint aligns with the first element of the input array.

    Index Shift: Because the offset is 2, the leftmost part of the mask corresponds to indices i−2i−2 through i+2i+2.

    For P[0]P[0]:
        N[−2]×M[0]N[−2]×M[0] (out of bounds, so ignored)
        N[−1]×M[1]N[−1]×M[1] (out of bounds, so ignored)
        N[0]×M[2]N[0]×M[2]
        N[1]×M[3]N[1]×M[3]
        N[2]×M[4]N[2]×M[4]

            M[2]=5 is multiplied with N[0]N[0].
    M[3]=4M[3]=4 is multiplied with N[1]N[1].
    M[4]=3M[4]=3 is multiplied with N[2]N[2].

So, the mask is applied in the order:

    N[0]×M[2]N[0]×M[2]
    N[1]×M[3]N[1]×M[3]
    N[2]×M[4]N[2]×M[4]


    Why This Order?

The mask is centered over the current position in the input array, so:

    M[2]M[2] is the "center" of the mask and corresponds to the current element N[0]N[0].
    M[3]M[3] and M[4]M[4] correspond to elements to the right of the center.
    M[0]M[0] and M[1]M[1] (if not out of bounds) would correspond to elements to the left.


    for P[1]= 4+10+12+12=38
    FOR P[2]=(1×3)+(2×4)+(3×5)+(4×4)+(5×3) = 3+8+15+16+15=57

*/    