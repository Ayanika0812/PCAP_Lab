#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char* argv[])
{
    printf("Reg_no is 220905128\n");    
    int rank,size,x=atoi(argv[1]);

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0)
    {
        MPI_Ssend(&x,1,MPI_INT,1,1,MPI_COMM_WORLD);
        printf("Sent %d to Process 1\n",x);
        MPI_Recv(&x,1,MPI_INT,size-1,1,MPI_COMM_WORLD,&status);
        printf("Received %d in Process %d\n",x,rank);
    }
    else
    {
        int t=(rank+1)%size;
        MPI_Recv(&x,1,MPI_INT,rank-1,1,MPI_COMM_WORLD,&status);
        printf("Received %d in process %d\n",x,rank);
        x++;
        MPI_Ssend(&x,1,MPI_INT,t,1,MPI_COMM_WORLD);
        printf("Sent %d to Process %d\n",x,t);
    }
    MPI_Finalize();


    return 0;
}

/*
 mpicc q4.c -o q4
student@lpcp-22:~/220905128/lab2$ mpirun -n 3 ./q4 4
Reg_no is 220905128
Reg_no is 220905128
Reg_no is 220905128
Sent 4 to Process 1
Received 4 in process 1
Sent 5 to Process 2
Received 5 in process 2
Received 6 in Process 0
Sent 6 to Process 0
*/