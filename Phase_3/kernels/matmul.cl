// Size of one work group block
# define TILE_SIZE 16

__kernel void matmul(__global const float* A, 
                     __global const float* B, 
                     __global float* C, 
                     int M, int N, int K, 
                     int is_B_transposed) {
    
    // Local IDs
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    // Global IDs
    int globalRow = get_global_id(1); // Row in C
    int globalCol = get_global_id(0); // Col in C

    // Shared memory tiles
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float value = 0.0f;
    
    // Loop over tiles
    int numberOfTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tileIndex = 0; tileIndex < numberOfTiles; tileIndex++) {
        
        // ---------------------------------------------------------
        // 1. LOAD TILE A
        // ---------------------------------------------------------
        int tiledCol = tileIndex * TILE_SIZE + localCol;
        
        if (globalRow < M && tiledCol < K) {
            tileA[localRow][localCol] = A[globalRow * K + tiledCol];
        } else {
            tileA[localRow][localCol] = 0.0f;
        }

        // ---------------------------------------------------------
        // 2. LOAD TILE B (CRITICAL FIX HERE)
        // ---------------------------------------------------------
        int tiledRow = tileIndex * TILE_SIZE + localRow;

        if (is_B_transposed) {
            // TRANSPOSED MODE (For Q * K^T)
            // We want B^T. Row 'r' of B^T is Column 'r' of B.
            // We need the element at B[globalCol][tiledRow]
            // Since B is (N x K) physically here, index is:
            if (globalCol < N && tiledRow < K) {
                // Notice: We read B, but indices are swapped relative to standard
                tileB[localRow][localCol] = B[globalCol * K + tiledRow]; 
            } else {
                tileB[localRow][localCol] = 0.0f;
            }
        } else {
            // STANDARD MODE (For Scores * V)
            // We read B normally.
            if (globalCol < N && tiledRow < K) {
                // ERROR FIX: You had "A[...]" here. Changed to "B[...]".
                tileB[localRow][localCol] = B[tiledRow * N + globalCol];
            } else {
                tileB[localRow][localCol] = 0.0f;
            }
        }

        // Sync to ensure tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute
        for (int i = 0; i < TILE_SIZE; i++) {
            value += tileA[localRow][i] * tileB[i][localCol];
        }

        // Sync before overwriting tile in next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write Result
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = value;
    }   
}