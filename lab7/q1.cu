#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

#define MAX_WORDS 1024  
#define WORD_LENGTH 32  

__device__ bool strcmp_cuda(const char *a, const char *b) {                      
    while (*a && (*a == *b)) {
        a++;
        b++;
    }
    return (*a == '\0' && *b == '\0');  
}

__global__ void count_word_kernel(char *words, int num_words, char *target, int *count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_words) return;

    char *word = words + idx * WORD_LENGTH;  
    if (strcmp_cuda(word, target)) {
        atomicAdd(count, 1);  
    }
}

int main() {
    char sentence[1024], target_word[32];

    // Take input for the sentence and target word
    printf("Enter the sentence: ");
    fgets(sentence, sizeof(sentence), stdin);
    sentence[strcspn(sentence, "\n")] = '\0';  // Remove newline character

    printf("Enter the word to search for: ");
    fgets(target_word, sizeof(target_word), stdin);
    target_word[strcspn(target_word, "\n")] = '\0';  // Remove newline character

    char h_words[MAX_WORDS * WORD_LENGTH] = {0}; 
    int num_words = 0;

    char temp_sentence[1024];
    strcpy(temp_sentence, sentence);
    char *token = strtok(temp_sentence, " ");

    // Split sentence into words
    while (token != NULL && num_words < MAX_WORDS) {
        strncpy(&h_words[num_words * WORD_LENGTH], token, WORD_LENGTH - 1);
        h_words[num_words * WORD_LENGTH + WORD_LENGTH - 1] = '\0'; // Null terminate
        num_words++;
        token = strtok(NULL, " ");
    }

    char *d_words, *d_target;
    int *d_count, h_count = 0;

    // Allocate device memory
    cudaMalloc((void**)&d_words, MAX_WORDS * WORD_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_target, WORD_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_words, h_words, MAX_WORDS * WORD_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_word, WORD_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (num_words + threads_per_block - 1) / threads_per_block;
    count_word_kernel<<<blocks, threads_per_block>>>(d_words, num_words, d_target, d_count);

    // Copy result from device to host
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Word '%s' appeared %d times in the sentence.\n", target_word, h_count);

    // Free device memory
    cudaFree(d_words);
    cudaFree(d_target);
    cudaFree(d_count);

    return 0;
}


/*
 ./q1
Enter the sentence: the cat sat on the bench and the bat was flying
Enter the word to search for: the
Word 'the' appeared 3 times in the sentence.
student@lpcp-19:~/220905128/lab7$ ./q1
Enter the sentence: hi hello hi 
Enter the word to search for: hi
Word 'hi' appeared 2 times in the sentence.
*/