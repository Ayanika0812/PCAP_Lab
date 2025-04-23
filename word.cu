#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define WORD_LEN 10
#define TOTAL_WORDS 4

__constant__ char d_searchWord[WORD_LEN];

// Kernel to search for the word
__global__ void searchWordKernel(char *d_words, int *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < TOTAL_WORDS) {
        char word[WORD_LEN];
        for (int i = 0; i < WORD_LEN; i++) {
            word[i] = d_words[idx * WORD_LEN + i];
        }

        // Compare word with search word in constant memory
        bool match = true;
        for (int i = 0; i < WORD_LEN; i++) {
            if (word[i] != d_searchWord[i]) {
                match = false;
                break;
            }
        }

        if (match) {
            *result = idx;
        }
    }
}

int main() {
    const char h_words[TOTAL_WORDS][WORD_LEN] = {"Apple", "Banana", "Mango", "Grape"};
    const char h_searchWord[WORD_LEN] = "Mango";

    char *d_words;
    int *d_result, h_result = -1;

    size_t size = TOTAL_WORDS * WORD_LEN * sizeof(char);
    cudaMalloc((void**)&d_words, size);
    cudaMemcpy(d_words, h_words, size, cudaMemcpyHostToDevice);

    // Copy search word to constant memory
    cudaMemcpyToSymbol(d_searchWord, h_searchWord, WORD_LEN * sizeof(char));

    cudaMalloc((void**)&d_result, sizeof(int));
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Kernel launch with 1D block and 1D grid
    searchWordKernel<<<1, 8>>>(d_words, d_result);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_result != -1) {
        printf("Word found at index %d\n", h_result);
    } else {
        printf("Word not found\n");
    }

    printf("Time taken: %f ms\n", milliseconds);

    // Free memory
    cudaFree(d_words);
    cudaFree(d_result);

    return 0;
}
