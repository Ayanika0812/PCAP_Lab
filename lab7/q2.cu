#include <stdio.h>
#include <cuda_runtime.h>

__global__ void generateRS(char *S, char *RS, int lenS) {
    int idx = threadIdx.x;
    if (idx < lenS) {
        int startIdx = (lenS * idx) - (idx * (idx - 1)) / 2;
        for (int i = 0; i < lenS - idx; i++) {
            RS[startIdx + i] = S[i];
        }
    }
}

int main() {
    char h_S[100];
    printf("Enter input string S: ");
    scanf("%s", h_S);
    
    int lenS = strlen(h_S);
    int lenRS = (lenS * (lenS + 1)) / 2;
    char *h_RS = (char*)malloc(lenRS + 1);
    h_RS[lenRS] = '\0';
    
    char *d_S, *d_RS;
    cudaMalloc((void**)&d_S, lenS * sizeof(char));
    cudaMalloc((void**)&d_RS, lenRS * sizeof(char));
    
    cudaMemcpy(d_S, h_S, lenS * sizeof(char), cudaMemcpyHostToDevice);
    
    generateRS<<<1, lenS>>>(d_S, d_RS, lenS);
    
    cudaMemcpy(h_RS, d_RS, lenRS * sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("Input string S: %s\n", h_S);
    printf("Output string RS: %s\n", h_RS);
    
    cudaFree(d_S);
    cudaFree(d_RS);
    free(h_RS);
    
    return 0;
}
