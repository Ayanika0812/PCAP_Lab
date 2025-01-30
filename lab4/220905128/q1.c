#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void ErrorHandler(int err_code, const char* context) {
    if (err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_err_string, err_class;
        MPI_Error_class(err_code, &err_class);
        MPI_Error_string(err_code, error_string, &length_err_string);
        printf("Error in %s: %d %s\n", context, err_class, error_string);
    }
}

int main(int argc, char* argv[]) {
    int rank, size, fact = 1, factsum, err_code;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ErrorHandler(err_code, "MPI_Comm_rank");

    // Get the size of the communicator
    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    ErrorHandler(err_code, "MPI_Comm_size");

    // Print the size and rank for debugging purposes
    printf("Size: %d, Rank: %d\n", size, rank);

    // Exit if there are no processes or only 1 process
    if (size <= 1) {
        printf("Error: MPI_COMM_SIZE is %d. Cannot run with zero or one process.\n", size);
        MPI_Finalize();
        return 1;
    }

    // Output process information
    printf("Process %d out of %d\n", rank, size);

    int sendVal = rank + 1;
    // Perform MPI_Scan to calculate the product
    err_code = MPI_Scan(&sendVal, &fact, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    ErrorHandler(err_code, "MPI_Scan (MPI_PROD)");

    // Print result after MPI_Scan (MPI_PROD)
    printf("Process %d received value %d after MPI_Scan (MPI_PROD)\n", rank, fact);

    // Perform another MPI_Scan for summing up the factorials
    err_code = MPI_Scan(&fact, &factsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ErrorHandler(err_code, "MPI_Scan (MPI_SUM)");

    // Only the last process prints the sum of all factorials
    if (rank == size - 1) {
        printf("Sum of all factorials till %d! = %d\n", rank + 1, factsum);
    }

    // Finalize MPI
    MPI_Finalize();

    // Print student registration number (debugging purpose)
    printf("Reg_no is 220905128\n");

    return 0;
}
/*
student@lpcp-19:~/220905128/lab4$ mpicc q12.c -o q12
student@lpcp-19:~/220905128/lab4$ mpirun -n 0 ./q12
Size: 1, Rank: 0
Error: MPI_COMM_SIZE is 1. Cannot run with zero or one process.
*/