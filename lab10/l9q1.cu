#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel function for Sparse Matrix-Vector multiplication
__global__ void sparseMatrixVectorMultiply(int *d_values, int *d_columns, int *d_row_ptr, int *d_vector, int *d_result, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        int start = d_row_ptr[row];
        int end = d_row_ptr[row + 1];
        int dot_product = 0;
        
        // Perform the dot product of the sparse row with the vector
        for (int i = start; i < end; i++) {
            dot_product += d_values[i] * d_vector[d_columns[i]];
        }
        
        d_result[row] = dot_product;
    }
}

int main() {
    int num_rows, num_cols, num_non_zero_elements;

    // Input: Matrix dimensions and number of non-zero elements
    printf("Enter number of rows in the matrix: ");
    scanf("%d", &num_rows);
    
    printf("Enter number of columns in the matrix: ");
    scanf("%d", &num_cols);
    
    printf("Enter the number of non-zero elements: ");
    scanf("%d", &num_non_zero_elements);
    
    int *values = (int*)malloc(num_non_zero_elements * sizeof(int));
    int *columns = (int*)malloc(num_non_zero_elements * sizeof(int));
    int *row_ptr = (int*)malloc((num_rows + 1) * sizeof(int));
    int *vector = (int*)malloc(num_cols * sizeof(int));
    int *result = (int*)malloc(num_rows * sizeof(int));

    // Input: Non-zero values, column indices, and row pointers
    printf("Enter the non-zero values of the matrix:\n");
    for (int i = 0; i < num_non_zero_elements; i++) {
        printf("Value %d: ", i + 1);
        scanf("%d", &values[i]);
    }

    printf("Enter the column indices for each non-zero value:\n");
    for (int i = 0; i < num_non_zero_elements; i++) {
        printf("Column index %d: ", i + 1);
        scanf("%d", &columns[i]);
    }

    printf("Enter the row pointers (length should be %d):\n", num_rows + 1);
    for (int i = 0; i <= num_rows; i++) {
        printf("Row pointer %d: ", i);
        scanf("%d", &row_ptr[i]);
    }

    // Input: Vector
    printf("Enter the input vector:\n");
    for (int i = 0; i < num_cols; i++) {
        printf("Vector element %d: ", i + 1);
        scanf("%d", &vector[i]);
    }

    // Allocate memory on the device
    int *d_values, *d_columns, *d_row_ptr, *d_vector, *d_result;
    cudaMalloc((void**)&d_values, num_non_zero_elements * sizeof(int));
    cudaMalloc((void**)&d_columns, num_non_zero_elements * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_vector, num_cols * sizeof(int));
    cudaMalloc((void**)&d_result, num_rows * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_values, values, num_non_zero_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, columns, num_non_zero_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, num_cols * sizeof(int), cudaMemcpyHostToDevice);

    // Define the grid and block sizes
    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    // Launch the kernel
    sparseMatrixVectorMultiply<<<numBlocks, blockSize>>>(d_values, d_columns, d_row_ptr, d_vector, d_result, num_rows);

    // Copy the result back to host
    cudaMemcpy(result, d_result, num_rows * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result of Sparse Matrix-Vector multiplication:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");

    // Free device and host memory
    cudaFree(d_values);
    cudaFree(d_columns);
    cudaFree(d_row_ptr);
    cudaFree(d_vector);
    cudaFree(d_result);

    free(values);
    free(columns);
    free(row_ptr);
    free(vector);
    free(result);

    return 0;
}


/*
./l9q1
Enter number of rows in the matrix: 3
Enter number of columns in the matrix: 3
Enter the number of non-zero elements: 3
Enter the non-zero values of the matrix:
Value 1: 1
Value 2: 2
Value 3: 3
Enter the column indices for each non-zero value:
Column index 1: 0
Column index 2: 1
Column index 3: 0
Enter the row pointers (length should be 4):
Row pointer 0: 0
Row pointer 1: 1
Row pointer 2: 2
Row pointer 3: 3
Enter the input vector:
Vector element 1: 4
Vector element 2: 5
Vector element 3: 6
Result of Sparse Matrix-Vector multiplication:
4 10 12 
*/