#include <stdio.h>
#include <cuda.h>

// 🔹 Basic kernel that just adds 1 to each matrix element
__global__ void processMatrix(int *d_A, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col;

    if (row < M && col < N) {
        d_A[idx] += 1;  // Dummy operation
    }
}

int main() {
    int M, N;
    printf("Enter rows (M) and columns (N): ");
    scanf("%d %d", &M, &N);

    int size = M * N * sizeof(int);
    int *A = (int *)malloc(size);

    printf("Enter elements of Matrix A:\n");
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &A[i]);
    }

    int *d_A;
    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    // 🔹 Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 🔸 Kernel call
    processMatrix<<<gridSize, blockSize>>>(d_A, M, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    printf("\nOutput Matrix (each element +1):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nTime taken: %f ms\n", milliseconds);

    cudaFree(d_A);
    free(A);
    return 0;
}

