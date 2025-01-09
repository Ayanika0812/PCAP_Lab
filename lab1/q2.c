#include "mpi.h"
#include<stdio.h>

int main(int argc, char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank % 2 == 0){
		printf("Process %d : Hello", rank);
	}
	else{
		printf("Process %d : World", rank);
	}
	printf("\n \n");
    MPI_Finalize();
    return 0;
}

/*
mpicc q2.c -o q2 
student@lpcp-22:~/220905128/lab1$ mpirun -n 6 ./q2
Process 1 : World
 
Process 4 : Hello
 
Process 0 : Hello
 
Process 2 : Hello
 
Process 3 : World
 
Process 5 : World
 
student@lpcp-22:~/220905128/lab1$ mpirun -n 3 ./q2
Process 0 : Hello
 
Process 1 : World
 
Process 2 : Hello
 
*/