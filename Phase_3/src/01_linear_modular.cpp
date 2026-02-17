#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

string loadkernel(const char* path){
    ifstream file(path);
    if (!file.is_open()) { cerr << "Missing Kernel: " << path << endl; exit(1); }
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

int main(){
    // --- 1. SETUP PLATFORM ---
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context,device);

    // --- 2. BUILD KERNELS (Modular Compilation) ---
    string srcMatMul = loadkernel("kernels/matmul.cl");
    string srcRelU = loadkernel("kernels/relu.cl");
    string srcSoftmax = loadkernel("kernels/softmax.cl"); // [NEW]

    cl::Program progMatMul(context, srcMatMul);
    if(progMatMul.build({device}) != CL_SUCCESS){
        cerr << "MatMul Build Error: " << progMatMul.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl; return 1;
    }

    cl::Program progRelU(context, srcRelU);
    if(progRelU.build({device}) != CL_SUCCESS){
        cerr << "ReLU Build Error: " << progRelU.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl; return 1;
    }

    cl::Program progSoftmax(context, srcSoftmax); // [NEW]
    if(progSoftmax.build({device}) != CL_SUCCESS){
        cerr << "Softmax Build Error: " << progSoftmax.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl; return 1;
    }

    // --- 3. DATA SHAPES (Batch=4, In=128, Out=64) ---
    int M = 4, K = 128, N = 64;

    vector<float> Input(M*K, 1.0f);
    vector<float> Weights(K*N, -0.5f);
    vector<float> Output(M*N, 0.0f);

    // --- 4. BUFFERS ---
    cl::Buffer d_Input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*Input.size(), Input.data());
    cl::Buffer d_Weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*Weights.size(), Weights.data());
    cl::Buffer d_Output(context, CL_MEM_READ_WRITE, sizeof(float)*Output.size(), nullptr);

    // --- 5. EXECUTION CHAIN ---
    // STEP A: Matrix Multiplication
    cl::Kernel k_matmul(progMatMul,"matmul"); // Ensure correct kernel name
    k_matmul.setArg(0,d_Input);
    k_matmul.setArg(1,d_Weights);
    k_matmul.setArg(2,d_Output);
    k_matmul.setArg(3, M);
    k_matmul.setArg(4, N);
    k_matmul.setArg(5, K);
    k_matmul.setArg(6, 0);

    //Padding for Tiled MatMul (16,16)
    int global_M = ((M+15/16))*16;
    int global_N = ((N+15/16))*16;

    cout << "1. Launching MatMul..." << endl;
    queue.enqueueNDRangeKernel(k_matmul, cl::NullRange, cl::NDRange(global_N, global_M), cl::NDRange(16,16));
    
    // [DEBUG] Print after MatMul
    queue.enqueueReadBuffer(d_Output, CL_TRUE, 0, sizeof(float)*Output.size(), Output.data());
    cout << "   -> MatMul Output[0]: " << Output[0] << " (Expected -64.0)" << endl;

    // STEP B: ReLU
    cl::Kernel k_relu(progRelU, "relu_activation");
    int total_elements = M * N;
    k_relu.setArg(0, d_Output);
    k_relu.setArg(1, total_elements);

    cout << "2. Launching ReLU..." << endl;
    queue.enqueueNDRangeKernel(k_relu, cl::NullRange, cl::NDRange(total_elements),cl::NullRange);

    // [DEBUG] Print after ReLU
    queue.enqueueReadBuffer(d_Output, CL_TRUE, 0, sizeof(float)*Output.size(), Output.data());
    cout << "   -> ReLU Output[0]: " << Output[0] << " (Expected 0.0)" << endl;

    // STEP C: Softmax [NEW]
    cl::Kernel k_softmax(progSoftmax, "softmax_row");
    k_softmax.setArg(0, d_Output);
    k_softmax.setArg(1, N);

    cout << "3. Launching Softmax..." << endl;
    queue.enqueueNDRangeKernel(k_softmax, cl::NullRange, cl::NDRange(M), cl::NullRange);

    // --- 6. READ BACK ---
    queue.enqueueReadBuffer(d_Output, CL_TRUE, 0, sizeof(float)*Output.size(), Output.data());

    cout << "Result[0]: " << Output[0] << " (Expected ~0.0156)" << endl;
    
    float sum = 0.0f;
    for(int i=0; i<N; i++) sum += Output[i];
    cout << "Sum of Row 0: " << sum << " (Expected 1.0)" << endl;

    if (Output[0] > 0.0f && abs(sum - 1.0f) < 0.001f) 
        cout << "SUCCESS: Pipeline Complete." << endl;
    else 
        cout << "FAILURE: Check logic." << endl;

    return 0;
}