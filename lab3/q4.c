#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char** argv)
{
    int rank,size,N,i,M,l=0;
    char str1[100];
    char str2[100];
    char B1[100];
    char C[200];
    char concat[100];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(rank==0)
    {   
        printf("I am Rank %d\n",rank);
        N = size;
        printf("Enter String 1 : \t");
        scanf("%[^\n]c",str1);
        printf("Enter String 2 : \t");
        scanf(" %[^\n]c",str2);
        M = strlen(str1)/N;
    }
    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);

    MPI_Scatter(str1,M,MPI_CHAR,B1,M,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Scatter(str2,M,MPI_CHAR,B1+M,M,MPI_CHAR,0,MPI_COMM_WORLD);

    l=0;
    for(i=0;i<M;i++)
    {
        concat[l++] = B1[i];
        concat[l++] = B1[i+M];
    }
        printf("I am Rank %d\n",rank);
    MPI_Gather(concat,2*M,MPI_CHAR,C,2*M,MPI_CHAR,0,MPI_COMM_WORLD);
    
    if(rank==0)
    {
        printf("Resultant String : %s\n",C);        
    }
    MPI_Finalize();
        printf("Reg_no 220905128\n");
    return 0;
}


/*

*/