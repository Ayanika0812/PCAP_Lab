#include "mpi.h"
#include<stdio.h>

long long factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main(int argc, char *argv[]){
	int rank, size;
	int fib[15] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377};
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);



    if(rank % 2 == 0){
		printf("Process %d - Fact %lld\n", rank, factorial(rank));
	}
	else{
		printf("Process %d - Fib %d\n", rank, fib[rank]);
	}
	printf("\n \n");
    MPI_Finalize();
    return 0;
}

/*

mpicc q5.c -o q5 
student@lpcp-22:~/220905128/lab1$ mpirun -n 6 ./q5
Process 2 - Fact 2

 
Process 3 - Fib 2

 
Process 0 - Fact 1

 
Process 1 - Fib 1

 
Process 4 - Fact 24

 
Process 5 - Fib 5
*/