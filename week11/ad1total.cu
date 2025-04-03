#include <stdio.h>
#include <cuda_runtime.h>

#define MAX_ITEMS 10
#define MAX_FRIENDS 100

typedef struct {
    float price;
} Item;

__global__ void calculatePurchase(float* purchases, int* itemIndices, const float* itemPrices, int numFriends, int numItems) {
    int friendId = blockIdx.x * blockDim.x + threadIdx.x;

    if (friendId < numFriends) {
        float total = 0.0f;

        for (int i = 0; i < numItems; i++) {
            int itemId = itemIndices[friendId * numItems + i];
            if (itemId >= 0 && itemId < numItems) {
                total += itemPrices[itemId];
            }
        }

        purchases[friendId] = total;
    }
}

int main() {
    Item items[MAX_ITEMS] = {
        {25.0f},
        {50.0f},
        {40.0f},
        {15.0f},
        {30.0f},
        {60.0f},
        {20.0f},
        {5.0f},
        {10.0f},
        {12.0f}
    };

    const int numItems = 10;
    int numFriends;

    printf("Enter the number of friends: ");
    scanf("%d", &numFriends);

    if (numFriends > MAX_FRIENDS) {
        printf("The number of friends can't exceed %d.\n", MAX_FRIENDS);
        return -1;
    }

    printf("\nShopping Mall Menu:\n");
    for (int i = 0; i < numItems; i++) {
        printf("%d. Item %d - $%.2f\n", i + 1, i + 1, items[i].price);
    }

    int* itemIndices = (int*)malloc(numFriends * numItems * sizeof(int));

    for (int i = 0; i < numFriends; i++) {
        printf("\nEnter the items bought by Friend %d (enter item numbers, 0 to stop):\n", i + 1);
        
        for (int j = 0; j < numItems; j++) {
            itemIndices[i * numItems + j] = -1;
        }

        for (int j = 0; j < numItems; j++) {
            printf("Item number to purchase (1 to %d, 0 to stop): ", numItems);
            int itemChoice;
            scanf("%d", &itemChoice);

            if (itemChoice == 0) break;

            itemIndices[i * numItems + j] = itemChoice - 1;
        }
    }

    float* d_purchases;
    int* d_itemIndices;
    float* d_itemPrices;

    cudaMalloc((void**)&d_purchases, numFriends * sizeof(float));
    cudaMalloc((void**)&d_itemIndices, numFriends * numItems * sizeof(int));
    cudaMalloc((void**)&d_itemPrices, numItems * sizeof(float));

    cudaMemset(d_purchases, 0, numFriends * sizeof(float));

    cudaMemcpy(d_itemIndices, itemIndices, numFriends * numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_itemPrices, &items[0].price, numItems * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numFriends + blockSize - 1) / blockSize;
    calculatePurchase<<<numBlocks, blockSize>>>(d_purchases, d_itemIndices, d_itemPrices, numFriends, numItems);

    float* purchases = (float*)malloc(numFriends * sizeof(float));
    cudaMemcpy(purchases, d_purchases, numFriends * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nTotal purchases made by each friend:\n");
    float totalPurchase = 0.0f;  // Variable to hold the total purchase by all friends

    for (int i = 0; i < numFriends; i++) {
        printf("Friend %d: $%.2f\n", i + 1, purchases[i]);
        totalPurchase += purchases[i];  // Add each friend's total to the overall total
    }

    printf("\nTotal purchases by all friends: $%.2f\n", totalPurchase);  // Display the total purchase

    cudaFree(d_purchases);
    cudaFree(d_itemIndices);
    cudaFree(d_itemPrices);

    free(itemIndices);
    free(purchases);

    return 0;
}




/*

student@lpcp-19:~/220905128/week11$ nvcc ad1.cu -o ad1
student@lpcp-19:~/220905128/week11$ ./ad1
Enter the number of friends: 2

Shopping Mall Menu:
1. Item 1 - $25.00
2. Item 2 - $50.00
3. Item 3 - $40.00
4. Item 4 - $15.00
5. Item 5 - $30.00
6. Item 6 - $60.00
7. Item 7 - $20.00
8. Item 8 - $5.00
9. Item 9 - $10.00
10. Item 10 - $12.00

Enter the items bought by Friend 1 (enter item numbers, 0 to stop):
Item number to purchase (1 to 10, 0 to stop): 2
Item number to purchase (1 to 10, 0 to stop): 3
Item number to purchase (1 to 10, 0 to stop): 0

Enter the items bought by Friend 2 (enter item numbers, 0 to stop):
Item number to purchase (1 to 10, 0 to stop): 2
Item number to purchase (1 to 10, 0 to stop): 4
Item number to purchase (1 to 10, 0 to stop): 5
Item number to purchase (1 to 10, 0 to stop): 0

Total purchases made by each friend:
Friend 1: $90.00
Friend 2: $95.00

Total purchases by all friends: $185.00
*/
