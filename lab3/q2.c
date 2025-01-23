#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, M, N;
    int *array = NULL, *sub_array = NULL;
    float local_avg = 0.0, total_avg = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Read M and calculate total number of elements
        printf("Enter M (number of elements per process): ");
        scanf("%d", &M);
        N = size; // Number of processes
        int total_elements = M * N;

        // Allocate and initialize the array
        array = (int *)malloc(total_elements * sizeof(int));
        printf("Enter %d elements:\n", total_elements);
        for (int i = 0; i < total_elements; i++) {
            scanf("%d", &array[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for sub-array
    sub_array = (int *)malloc(M * sizeof(int));

    // Scatter the array elements
    MPI_Scatter(array, M, MPI_INT, sub_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the local average
    int local_sum = 0;
    for (int i = 0; i < M; i++) {
        local_sum += sub_array[i];
        printf("I am Rank %d, received number: %d, Local_sum: %d\n", rank, sub_array[i], local_sum);    
    }
    local_avg = (float)local_sum / M;
    
    // Reduce to calculate the total average
    MPI_Reduce(&local_avg, &total_avg, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    printf("I am Rank %d, Local_Avg: %f\n", rank, local_avg);   
    if (rank == 0) {
        total_avg /= N; // Divide by the number of processes
        printf("Total average: %.2f\n", total_avg);
        printf("I am Rank %d, Total_Avg: %f\n", rank, total_avg);       
    }

    // Free allocated memory
    if (rank == 0) {
        free(array);
    }
    free(sub_array);

    MPI_Finalize();
    return 0;
}

/*
mpicc q2.c -o q2
student@lpcp-19:~/220905128/lab3$ mpirun -n 2 ./q2
Enter M (number of elements per process): 2
Enter 4 elements:
1
3 2 4
I am Rank 0, received number: 1, Local_sum: 1
I am Rank 0, received number: 3, Local_sum: 4
I am Rank 0, Local_Avg: 2.000000
Total average: 2.50
I am Rank 0, Total_Avg: 2.500000
I am Rank 1, received number: 2, Local_sum: 2
I am Rank 1, received number: 4, Local_sum: 6
I am Rank 1, Local_Avg: 3.000000

*/