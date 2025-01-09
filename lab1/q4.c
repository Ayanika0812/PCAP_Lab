#include "mpi.h"
#include<stdio.h>
#include<string.h>

char toggle(char c){

	if ( c >= 'A' && c<= 'Z'){
		return c+32;
			}
	else if(c >= 'a' && c<='z'){
		return c-32;
	}
	return c;
}

int main(int argc, char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char c[20] = "HELlo";
	int c_length =strlen(c);
 /*if(rank < c_length){
     char s = c[rank];
     char toggled = toggle(s);
     printf("Rank %d toggled %c to %c\n  , The string is %s \n", rank,s,toggled,c);

     c[rank]=toggled;	
 }*/
    if (rank < c_length) {
        char s = c[rank];
        char toggled = toggle(s);
        c[rank] = toggled; // Update the local copy of the string
        
        // Print the character toggled by the current rank and the current state of the string
        printf("Rank %d toggled '%c' to '%c'\nThe string is: %s\n", rank, s, toggled, c);
    }


 
   
MPI_Finalize();

return 0;

}

/*
mpicc q4.c -o q4 
student@lpcp-22:~/220905128/lab1$ mpirun -n 5 ./q4
Rank 0 toggled 'H' to 'h'
The string is: hELlo
Rank 1 toggled 'E' to 'e'
The string is: HeLlo
Rank 3 toggled 'l' to 'L'
The string is: HELLo
Rank 2 toggled 'L' to 'l'
The string is: HEllo
Rank 4 toggled 'o' to 'O'
The string is: HELlO



*/