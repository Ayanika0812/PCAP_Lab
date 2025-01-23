#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int factorial(int n) {
    if (n == 0 || n == 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char* argv[]) {
    printf("Reg_no 220905128\n");
    int rank, size, N;
     
    MPI_Init(&argc, &argv);                    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    MPI_Comm_size(MPI_COMM_WORLD, &size);     

    int *numbers = NULL;                         
    int number;                               
    int factorial_result;                       
    int *factorials = NULL;                     

    if (rank == 0) {
        printf("I am Rank %d\n",rank);
        printf("Enter the number of values (N): ");
        scanf("%d", &N);

        if (N != size) {
            printf("Error: Number of processes must be equal to N.\n");
        }

        if (N != size) {
               if (rank == 0) {
                 printf("Error: Number of processes must be equal to N.\n");
                }
          MPI_Finalize(); // Finalize MPI before exiting
          exit(0);        // Exit the program safely
        }
          else {
            // Allocate memory for numbers
            numbers = (int *)malloc(N * sizeof(int));
            printf("Enter %d values:\n", N);
            for (int i = 0; i < N; i++) {
                scanf("%d", &numbers[i]);
            }

            // Allocate memory for factorial results
            factorials = (int *)malloc(N * sizeof(int));
        }
    }

    // Distribute one number to each process
    MPI_Scatter(numbers, 1, MPI_INT, &number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("I am Rank %d\n",rank);
    // Check for termination signal (-1)
    if (number == -1) {
        if (rank == 0) {
            free(numbers);    // Free allocated memory
            if (factorials != NULL) free(factorials);
        }
        MPI_Finalize(); // Gracefully exit
        return 0;
    }

    // Calculate factorial
    factorial_result = factorial(number);

    // Gather factorials at root process
    MPI_Gather(&factorial_result, 1, MPI_INT, factorials, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate the sum of factorials
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += factorials[i];
        }

        printf("Sum of factorials: %d\n", sum);

        free(numbers);    // Free allocated memory
        free(factorials);
    }

    MPI_Finalize();   // Finalize MPI
    
    return 0;
}


