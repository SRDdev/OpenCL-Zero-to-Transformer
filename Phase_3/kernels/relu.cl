/* Relu : This kernel contains the implemtation for relu activation function. 
Relu -> {0 if x < 0; x if x > 0}
*/

__kernel void relu_activation(__global float* data, const int N){
    //---------------------------------//
    //Get Data
    //---------------------------------//
    int id = get_global_id(0);

    //---------------------------------//
    // Boundary Check
    //---------------------------------//
    if(id < N){
        float val = data[id];
        data[id] = fmax(0.0f, val);
    }

}