#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void CheckProcessCount(int size, int rank) {
    if(size != 3) {
        if(rank == 0) {
            printf("Error: This program requires exactly 3 processes. You have %d processes.\n", size);
        }
        MPI_Finalize();
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    int rank, size, ele, result;
    int mat[3][3];

    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    CheckProcessCount(size, rank);
    
    if(rank == 0) {
        printf("Enter the elements in 3x3 matrix:\n");
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                scanf("%d", &mat[i][j]);
        printf("Enter element to be searched: ");
        scanf("%d", &ele);
    }

    int arr[3];
    MPI_Bcast(&ele, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(mat, 3, MPI_INT, arr, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int res = 0;
    for(int i = 0; i < 3; i++)
        if(arr[i] == ele)
            res++;
    MPI_Reduce(&res, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        printf("Total number of occurrences is: %d\n", result);
    }

    MPI_Finalize();
    printf("Reg_no is 220905128\n");
    exit(0);
}
/*
student@lpcp-19:~/220905128/lab4$ mpicc q2.c -o q2
student@lpcp-19:~/220905128/lab4$ mpirun -n 3 ./q2
Enter the elements in 3x3 matrix:
1 2 3 
3 2 1
1 3 2
Enter element to be searched: 3
Total number of occurrences is: 3
Reg_no is 220905128
Reg_no is 220905128
Reg_no is 220905128
*/