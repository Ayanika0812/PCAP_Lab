#include "mpi.h"
#include<stdio.h>

int main(int argc, char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	int x=9,y=2;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    int res =0;
    switch(rank){


    case 0: res = x+y;
	printf("The result of addition is %d from rank %d\n ",res , rank);
	break;

    case 1: res = x-y;
	printf("The result of subtraction is %d from rank %d\n ",res , rank);
	break;

    case 2: res = x*y;
	printf("The result of multiplication is %d from rank %d\n ",res , rank);
	break;

    case 3: res = x/y;
	printf("The result of division is %d from rank %d\n ",res , rank);
	break;

    }

    MPI_Finalize();

    return 0;
}

/*
mpicc q3.c -o q3 
student@lpcp-22:~/220905128/lab1$ mpirun -n 4 ./q3
The result of addition is 11 from rank 0
 The result of multiplication is 18 from rank 2
 The result of subtraction is 7 from rank 1
 The result of division is 4 from rank 3
*/