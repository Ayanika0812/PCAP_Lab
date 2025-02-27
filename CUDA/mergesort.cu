#include <stdio.h>
#include <cuda_runtime.h>

__device__ void merge(int *arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = new int[n1];
    int *R = new int[n2];

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

// Kernel function to perform merge sort
__global__ void mergeSortKernel(int *arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Sort first half
        mergeSortKernel<<<1, 1>>>(arr, left, mid);

        // Sort second half
        mergeSortKernel<<<1, 1>>>(arr, mid + 1, right);

        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

// Host function to initiate the merge sort
void mergeSort(int *arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to perform merge sort
    mergeSortKernel<<<1, 1>>>(d_arr, 0, n - 1);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int arr[] = {3, 2, 5, 1, 4};
    int n = sizeof(arr) / sizeof(arr[0]);

    mergeSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
