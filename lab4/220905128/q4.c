#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, err_code;
    char str[100];
    char resultant[1000]; 

    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the string: \n");
        scanf("%[^\n]c", str);
        if (size != strlen(str)) {
            printf("Error: This program requires the number of processes = length of the string.\n");
            exit(1);
        }
    }

    char rcvbuf[2];
    MPI_Scatter(str, 1, MPI_CHAR, rcvbuf, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    char modified_str[100] = {0}; // Buffer to store the repeated character string

    for (int i = 0; i < rank + 1; i++) {
        modified_str[i] = rcvbuf[0];
    }
    modified_str[rank + 1] = '\0';

    char temp_result[1000];
    MPI_Gather(modified_str, 100, MPI_CHAR, temp_result, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        resultant[0] = '\0';
        for (int i = 0; i < size; i++) {
            strcat(resultant, &temp_result[i * 100]);  // Append each process's result to the final string
        }
        printf("The final result is: %s\n", resultant);
    }

    MPI_Finalize();
    printf("Reg_no is 220905128\n");
    exit(0);
}
/*
student@lpcp-19:~/220905128/lab4$ mpicc q4.c -o q4
student@lpcp-19:~/220905128/lab4$ mpirun -n 4 ./q4
Enter the string: 
PCAP
The final result is: PCCAAAPPPP
Reg_no is 220905128
Reg_no is 220905128
Reg_no is 220905128
Reg_no is 220905128
*/