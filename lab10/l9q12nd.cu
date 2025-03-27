#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void spmvCSR(int *d_values, int *d_columns, int *d_row_ptr, int *d_vector, int *d_result, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int start = d_row_ptr[row];
        int end = d_row_ptr[row + 1];
        int sum = 0;
        for (int j = start; j < end; j++) {
            sum += d_values[j] * d_vector[d_columns[j]];
        }
        d_result[row] = sum;
    }
}

int main() {
    int num_rows, num_cols, num_nonzeros;
    printf("Enter number of rows in the matrix: ");
    scanf("%d", &num_rows);
    printf("Enter number of columns in the matrix: ");
    scanf("%d", &num_cols);
    printf("Enter number of non-zero elements: ");
    scanf("%d", &num_nonzeros);
    int *h_values   = (int*)malloc(num_nonzeros * sizeof(int));
    int *h_columns  = (int*)malloc(num_nonzeros * sizeof(int));
    int *h_row_ptr  = (int*)malloc((num_rows + 1) * sizeof(int));
    int *h_vector   = (int*)malloc(num_cols * sizeof(int));
    int *h_result   = (int*)malloc(num_rows * sizeof(int));
    printf("Enter %d non-zero values: ", num_nonzeros);
    for (int i = 0; i < num_nonzeros; i++) {
        scanf("%d", &h_values[i]);
    }
    printf("Enter %d column indices (0-indexed): ", num_nonzeros);
    for (int i = 0; i < num_nonzeros; i++) {
        scanf("%d", &h_columns[i]);
    }
    printf("Enter %d row pointers: ", num_rows + 1);
    for (int i = 0; i <= num_rows; i++) {
        scanf("%d", &h_row_ptr[i]);
    }

    printf("Enter %d vector elements: ", num_cols);
    for (int i = 0; i < num_cols; i++) {
        scanf("%d", &h_vector[i]);
    }

    int *d_values, *d_columns, *d_row_ptr, *d_vector, *d_result;
    cudaMalloc((void**)&d_values, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_columns, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_vector, num_cols * sizeof(int));
    cudaMalloc((void**)&d_result, num_rows * sizeof(int));
    cudaMemcpy(d_values, h_values, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, h_columns, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, num_cols * sizeof(int), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;
    spmvCSR<<<numBlocks, blockSize>>>(d_values, d_columns, d_row_ptr, d_vector, d_result, num_rows);
    cudaMemcpy(h_result, d_result, num_rows * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result of Sparse Matrix-Vector multiplication:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%d ", h_result[i]);
    }
    printf("\n");
    cudaFree(d_values);
    cudaFree(d_columns);
    cudaFree(d_row_ptr);
    cudaFree(d_vector);
    cudaFree(d_result);
    free(h_values);
    free(h_columns);
    free(h_row_ptr);
    free(h_vector);
    free(h_result);

    return 0;
}

/*
/lab9q1
Enter number of rows in the matrix: 3
Enter number of columns in the matrix: 4
Enter number of non-zero elements: 4
Enter 4 non-zero values: 3 4 5 7
Enter 4 column indices (0-indexed): 0 3 2 1
Enter 4 row pointers: 0 2 3 4
Enter 4 vector elements: 1 2 3 4
Result of Sparse Matrix-Vector multiplication:
19 15 14
*/