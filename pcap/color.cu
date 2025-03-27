#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Include stb_image and stb_image_write
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// CUDA kernel to invert the colors
__global__ void invertColors(unsigned char *image, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        int pixelIndex = (idy * width + idx) * channels;

        // Invert each channel (R, G, B)
        for (int c = 0; c < channels; ++c) {
            image[pixelIndex + c] = 255 - image[pixelIndex + c];
        }
    }
}

int main() {
    // Load the image using stb_image
    int width, height, channels;
    unsigned char *h_image = stbi_load("image.png", &width, &height, &channels, 0); // Path is "image.png" now

    if (h_image == NULL) {
        printf("Error loading image!\n");
        return -1;
    }

    printf("Image loaded: %dx%d, %d channels\n", width, height, channels);

    // Allocate memory for image on the device
    unsigned char *d_image;
    cudaMalloc((void**)&d_image, width * height * channels * sizeof(unsigned char));

    // Copy the image data from host to device
    cudaMemcpy(d_image, h_image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set up kernel execution configuration
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);  // Number of blocks

    // Launch the kernel to invert the image colors
    invertColors<<<numBlocks, threadsPerBlock>>>(d_image, width, height, channels);

    // Check for kernel launch errors
    cudaDeviceSynchronize();

    // Copy the result back to host memory
    cudaMemcpy(h_image, d_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the inverted image
    if (!stbi_write_jpg("inverted_image.jpg", width, height, channels, h_image, 100)) {
        printf("Error saving the image!\n");
    }

    printf("Inverted image saved as 'inverted_image.jpg'\n");

    // Free device memory
    cudaFree(d_image);

    // Free host memory
    stbi_image_free(h_image);

    return 0;
}
