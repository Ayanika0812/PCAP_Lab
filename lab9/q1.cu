#include <stdio.h>
#include <cuda.h>

__global__ void modifyMatrix(int *A, int *B, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
                B[index] = A[index];
            } else {
                B[index] = ~A[index] & 0xFFFFFFFF; // Take 1's complement and mask out leading bits
            }
        }
    }
}

void printMatrix(int *B, int rows, int cols) {
    printf("\nModified Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = i * cols + j;
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                // For border elements, print the number directly
                printf("%d ", B[index]);
            } else {
                int num = B[index];
                int mask = 0xF;  // Mask to get the last 4 bits
                int last4Bits = num & mask;
                
                // Print only the last 4 bits in binary
                for (int k = 3; k >= 0; k--) {
                    printf("%d", (last4Bits >> k) & 1);
                }
                printf(" ");
            }
        }
        printf("\n");
    }
}

int main() {
    int M, N;
    printf("Enter the number of rows (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns (N): ");
    scanf("%d", &N);

    int h_A[M * N], h_B[M * N];
    printf("Enter matrix A elements in a single line (row-wise):\n");

    for (int i = 0; i < M * N; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A, *d_B;
    size_t size = M * N * sizeof(int);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    modifyMatrix<<<(M + 255) / 256, 256>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    printMatrix(h_B, M, N);

    return 0;
}


/*

student@lpcp-19:~/220905128/lab9$ nvcc q3.cu -o q3
student@lpcp-19:~/220905128/lab9$ ./q3
Enter the number of rows (M): 4
Enter the number of columns (N): 4
Enter matrix A elements in a single line (row-wise):
1 2 3 4
6 5 8 3
2 4 10 1
9 1 2 5

Modified Matrix:
1 2 3 4 
6 1010 0111 3 
2 1011 0101 1 
9 1 2 5 

*/