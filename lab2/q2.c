#include <mpi.h>
#include <stdio.h>

int main(int argc,char* argv[])
{   
    int rank,size,x;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;
    if(rank==0)
    {
        printf("Enter a number in Master Process :");
        scanf("%d",&x);
        for(int i=1;i<size;i++)
        {
            MPI_Send(&x,1,MPI_INT,i,1,MPI_COMM_WORLD);
            printf("Sent %d from process 0\n",x);
        }
            printf("Reg_no is 220905128\n");
    }
    else
    {
        MPI_Recv(&x,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
        printf("Received %d in process %d\n",x,rank);
    }

    MPI_Finalize();
    return 0;
}

/*
mpicc q2.c -o q2
student@lpcp-22:~/220905128/lab2$ mpirun -n 5 ./q2
Enter a number in Master Process :3
Sent 3 from process 0
Sent 3 from process 0
Sent 3 from process 0
Sent 3 from process 0
Received 3 in process 1
Received 3 in process 2
Received 3 in process 3
Received 3 in process 4
*/