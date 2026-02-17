__kernel void softmax_row(__global float* data, int N) {
    // 1. Which row am I responsible for?
    int row_idx = get_global_id(0);
    
    // Pointer arithmetic: Move to the start of my row
    __global float* my_row = data + (row_idx * N);

    // --- STEP A: Find Max (for numerical stability) ---
    // e^1000 explodes, but e^(1000-1000) = e^0 = 1. Safe.
    float max_val = -INFINITY;
    for (int i = 0; i < N; i++) {
        float val = my_row[i];
        if (val > max_val) max_val = val;
    }

    // --- STEP B: Exponentiate and Sum ---
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float exp_val = exp(my_row[i] - max_val);
        my_row[i] = exp_val; // Store it back temporarily
        sum += exp_val;
    }

    // --- STEP C: Normalize ---
    for (int i = 0; i < N; i++) {
        my_row[i] /= sum;
    }
}