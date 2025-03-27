#include <stdio.h>
#include <stdlib.h>

void apply_convolution(float* input, int input_size, float* mask, int mask_size, float* result) {
    int i, j;
    int mask_half = mask_size / 2;

    // Apply convolution (assuming zero-padding)
    for (i = 0; i < input_size; i++) {
        result[i] = 0.0f;

        // Apply the mask on the input array
        for (j = 0; j < mask_size; j++) {
            int index = i + j - mask_half; // calculate corresponding index in input array

            // Ensure we don't go out of bounds (zero padding)
            if (index >= 0 && index < input_size) {
                result[i] += input[index] * mask[j];
            }
        }
    }
}

int main() {
    int N_size, M_size;

    // Get the size of the input array
    printf("Enter the size of the input array N: ");
    scanf("%d", &N_size);
    
    // Dynamically allocate memory for the input array
    float *N = (float *)malloc(N_size * sizeof(float));
    
    // Get the input array values from the user
    printf("Enter the values of the input array N:\n");
    for (int i = 0; i < N_size; i++) {
        scanf("%f", &N[i]);
    }

    // Get the size of the mask array
    printf("Enter the size of the mask array M: ");
    scanf("%d", &M_size);
    
    // Dynamically allocate memory for the mask array
    float *M = (float *)malloc(M_size * sizeof(float));
    
    // Get the mask values from the user
    printf("Enter the values of the mask array M:\n");
    for (int i = 0; i < M_size; i++) {
        scanf("%f", &M[i]);
    }

    // Result array to store the convolution output
    float *P = (float *)malloc(N_size * sizeof(float));

    // Apply the convolution operation
    apply_convolution(N, N_size, M, M_size, P);

    // Print the result array
    printf("Result array P (after convolution):\n");
    for (int i = 0; i < N_size; i++) {
        printf("%f ", P[i]);
    }
    printf("\n");

    // Free dynamically allocated memory
    free(N);
    free(M);
    free(P);

    return 0;
}

/*
nvcc ad2.cu -o adq2
student@lpcp-19:~/220905128/lab10$ ./adq2
Enter the size of the input array N: 3
Enter the values of the input array N:
1 2 3
Enter the size of the mask array M: 3
Enter the values of the mask array M:
4 6 7
Result array P (after convolution):
20.000000 37.000000 26.000000 
student@lpcp-19:~/220905128/lab10$ ./adq2
Enter the size of the input array N: 5
Enter the values of the input array N:
2 5 7 4 2
Enter the size of the mask array M: 3
Enter the values of the mask array M:
3 1 4
Result array P (after convolution):
22.000000 39.000000 38.000000 33.000000 14.000000 
*/