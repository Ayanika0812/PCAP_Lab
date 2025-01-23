#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <ctype.h>

int is_vowel(char c) {
    c = tolower(c);
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int main(int argc, char *argv[]) {
    int rank, size;
    char *input_string = NULL;
    char *local_string = NULL;
    int local_count_vowels = 0, local_count_non_vowels = 0, total_count_vowels = 0, total_count_non_vowels = 0;
    int string_length, chunk_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        input_string = (char *)malloc(1000 * sizeof(char));
        

        printf("Enter a string: ");
        fgets(input_string, 1000, stdin);
        input_string[strcspn(input_string, "\n")] = 0;

        string_length = strlen(input_string);

     
    }

    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    chunk_size = string_length / size;

    local_string = (char *)malloc(chunk_size * sizeof(char));

    MPI_Scatter(input_string, chunk_size, MPI_CHAR, local_string, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("I am Rank %d, received : %s\n", rank, local_string);

    for (int i = 0; i < chunk_size; i++) {
        if (is_vowel(local_string[i])) {
            local_count_vowels++;
        } else {
            local_count_non_vowels++;
        }
    }

    MPI_Reduce(&local_count_vowels, &total_count_vowels, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count_non_vowels, &total_count_non_vowels, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total number of vowels: %d\n", total_count_vowels);
        printf("Total number of non-vowels: %d\n", total_count_non_vowels);
    }

    printf("Process %d found %d vowels and %d non-vowels.\n", rank, local_count_vowels, local_count_non_vowels);

    free(local_string);
    if (rank == 0) {
        free(input_string);
    }

    MPI_Finalize();

    printf("Reg_no 220905128\n");
    return 0;
}


/*
mpicc q3.c -o q3
student@lpcp-19:~/220905128/lab3$ mpirun -n 2 ./q3
Enter a string: programmings
I am Rank 0, received : progra
Total number of vowels: 3
Total number of non-vowels: 9
Process 0 found 2 vowels and 4 non-vowels.
I am Rank 1, received : mmings
Process 1 found 1 vowels and 5 non-vowels.
Reg_no 220905128
Reg_no 220905128

*/