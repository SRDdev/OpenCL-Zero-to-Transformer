#define TILE_SIZE 16

__kernel void mm_tiled(__global const float* A, 
                       __global const float* B, 
                       __global float* C, 
                       int M, 
                       int N, 
                       int K) {
    
    // Fast On-Chip Cache
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        
        // Load Tile A
        if (row < M && (t * TILE_SIZE + localCol) < K)
            tileA[localRow][localCol] = A[row * K + t * TILE_SIZE + localCol];
        else
            tileA[localRow][localCol] = 0.0f;

        // Load Tile B
        if (col < N && (t * TILE_SIZE + localRow) < K)
            tileB[localRow][localCol] = B[(t * TILE_SIZE + localRow) * N + col];
        else
            tileB[localRow][localCol] = 0.0f;

        // Sync: Wait for the whole team to finish loading tiles
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: Each thread uses the shared tiles
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[localRow][i] * tileB[i][localCol];
        }

        // Sync: Wait for the whole team to finish math before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}