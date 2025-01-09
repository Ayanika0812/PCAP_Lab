#include "mpi.h"
#include<stdio.h>
#include<math.h>

int main(int argc,char *argv[])
{
	int rank,size;

	MPI_Init(&argc,&argv); 
    
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    int x = size;
    
    printf("My rank is %d \n",rank);

    int res = pow(x,rank);
    printf("%d Power %d is %d\n", x, rank,res);
    
    MPI_Finalize();

    return 0;
}


/*

FOR MATH.H   -lm reqd for compilation
mpicc q1.c -o q1 -lm
student@lpcp-22:~/220905128/lab1$ mpirun -n 4 ./q1
My rank is 0 
4 Power 0 is 1
My rank is 1 
4 Power 1 is 4
My rank is 2 
4 Power 2 is 16
My rank is 3 
4 Power 3 is 64
*/