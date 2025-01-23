#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to calculate the factorial of a number
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
    int *numbers = NULL;     // Array to store input numbers (on rank 0)
    int number;              // Number received by each process
    int factorial_result;    // Factorial of the number computed by each process
    int *factorials = NULL;  // Array to store all factorials (on rank 0)

    MPI_Init(&argc, &argv);                  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);    // Get total number of processes

    if (rank == 0) {
        // Input the number of values (N) and ensure it matches the number of processes
        printf("Enter the number of values (N): ");
        scanf("%d", &N);

        if (N != size) {
            printf("Error: Number of processes must be equal to N.\n");
            MPI_Finalize();
            exit(0);
        }

        // Allocate memory for the numbers array
        numbers = (int *)malloc(N * sizeof(int));
        printf("Enter %d values:\n", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &numbers[i]);
        }

        // Allocate memory for the factorials array
        factorials = (int *)malloc(N * sizeof(int));
    }

    // Scatter the numbers to all processes
    MPI_Scatter(numbers, 1, MPI_INT, &number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the factorial of the received number
    factorial_result = factorial(number);
    printf("I am Rank %d, received number: %d, factorial: %d\n", rank, number, factorial_result);

    // Gather the factorials at the root process
    MPI_Gather(&factorial_result, 1, MPI_INT, factorials, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate the sum of all factorials
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += factorials[i];
        }

        printf("Sum of factorials: %d\n", sum);

        // Free allocated memory
        free(numbers);
        free(factorials);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}


/*
 mpicc q1.c -o q1
student@lpcp-19:~/220905128/lab3/lab3$ mpirun -n 2 ./q1
Reg_no 220905128
Reg_no 220905128
Enter the number of values (N): 2
Enter 2 values:
2
3
I am Rank 0, received number: 2, factorial: 2
Sum of factorials: 8
I am Rank 1, received number: 3, factorial: 6

*/
