#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void ErrorHandler(int err_code) {
    if(err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_err_string, err_class;
        MPI_Error_class(err_code, &err_class);
        MPI_Error_string(err_code, error_string, &length_err_string);
        printf("Error: %d %s\n", err_class, error_string);
    }
}
int main(int argc, char* argv[]) {
    int rank, size, fact = 1, factsum, i, err_code;
    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0){
        ErrorHandler(err_code);
    }
    int sendVal = rank + 1;
    err_code = MPI_Scan(&sendVal, &fact, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    if(rank == 0){
        ErrorHandler(err_code);
    }
    printf("Process %d received value %d after MPI_Scan (MPI_PROD)\n", rank, fact);
    err_code = MPI_Scan(&fact, &factsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0){
        ErrorHandler(err_code);
    }
    if(rank == size - 1){
        printf("Sum of all factorials till %d! = %d\n", rank + 1, factsum);
    }
    MPI_Finalize();
    printf("Reg_no is 220905128\n");
}

/*
student@lpcp-19:~/220905128/lab4$ mpicc q1.c -o q1
student@lpcp-19:~/220905128/lab4$ mpirun -n 3 ./q1
Process 0 received value 1 after MPI_Scan (MPI_PROD)
Process 2 received value 6 after MPI_Scan (MPI_PROD)
Process 1 received value 2 after MPI_Scan (MPI_PROD)
Sum of all factorials till 3! = 9
Reg_no is 220905128
Reg_no is 220905128
Reg_no is 220905128
*/