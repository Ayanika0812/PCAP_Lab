#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char*argv[])
{
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    MPI_Status status ,st1, st2;
    if(rank==0)
    {
       char data[10];  
       printf("Enter word: ");  
       scanf("%s",data); 
       int len = strlen(data);
       MPI_Ssend(&len,1,MPI_INT,1,0,MPI_COMM_WORLD);

       MPI_Ssend(data,len,MPI_CHAR,1,1,MPI_COMM_WORLD);       

       printf("Process %d sent the word %s\n ",rank, data);
       printf("Reg_no is 220905128\n");
       
       MPI_Recv(&data,len,MPI_CHAR,1,1,MPI_COMM_WORLD,&st2);
       printf("Toggled data received by P1 is %s \n",data);


    }
     
    else{
        int len;
        char data[10];
        MPI_Recv(&len,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);

        MPI_Recv(data,len,MPI_CHAR,0,1,MPI_COMM_WORLD,&st2);
        data[len]='\0';
        printf("Data received from P0 is %s\n ", data);

        for(int i=0;i<len;i++)
        {
            if(data[i]>=65 && data[i]<=90)
                data[i]= data[i]+ 32;

            else if(data[i]>=97 && data[i]<=122)
                data[i]= data[i]- 32;
        }

        MPI_Ssend(data,len,MPI_CHAR,0,1,MPI_COMM_WORLD);

    }
    MPI_Finalize();

       
    return 0;
}

/*
mpicc q1.c -o q1
student@lpcp-22:~/220905128/lab2$ mpirun -n 2 ./q1
Reg_no is  220905128
Reg_no is  220905128
Enter word: HEllo
Process 0 sent the word HEllo
 Data received from P0 is HEllo
 Toggled data received by P1 is heLLO
 */