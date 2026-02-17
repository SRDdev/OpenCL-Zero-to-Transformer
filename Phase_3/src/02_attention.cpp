#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
using namespace std;

string loadkernel(const char* path){
    ifstream file(path);
    if (!file.is_open()) {cerr << "Missing Kernel : " << path << endl; exit(1);}
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

int main(){
    // SETUP
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context,device);

    // KERNELS
    // Check your .cl files! If the function is "mm_robust", use that name.
    cl::Program progMatMul(context, loadkernel("kernels/matmul.cl"));
    if(progMatMul.build({device}) != CL_SUCCESS) {
        cerr << progMatMul.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl; return 1;
    }

    cl::Program progSoftMax(context, loadkernel("kernels/softmax.cl"));
    if(progSoftMax.build({device}) != CL_SUCCESS) {
        cerr << progSoftMax.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl; return 1;
    }

    int L = 4;
    int D = 64;

    cout << "Self-Attention: Sequence Length=" << L << ", Embed Dim=" << D << endl;

    vector<float> Q(L*D, 1.0f);
    vector<float> K_mat(L*D, 1.0f);
    vector<float> V(L*D, 2.0f);
    vector<float> Scores(L*L, 0.0f);
    vector<float> Output(L*D, 0.0f);

    cl::Buffer d_Q(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*Q.size(), Q.data());
    cl::Buffer d_K(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*K_mat.size(), K_mat.data());
    cl::Buffer d_V(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*V.size(), V.data());
    cl::Buffer d_Score(context, CL_MEM_READ_WRITE, sizeof(float)*Scores.size(), nullptr);
    cl::Buffer d_Output(context, CL_MEM_READ_WRITE, sizeof(float)*Output.size(), nullptr);

    // === STEP A: MatMul 1 (Q * K^T) ===
    // If your kernel function name is "mm_robust", change "matmul" to "mm_robust"
    cl::Kernel matmul_k(progMatMul, "matmul"); 
    matmul_k.setArg(0,d_Q);
    matmul_k.setArg(1,d_K);
    matmul_k.setArg(2,d_Score); 
    matmul_k.setArg(3,L); 
    matmul_k.setArg(4,L); 
    matmul_k.setArg(5,D); 
    matmul_k.setArg(6,1); 

    int g_L = ((L + 15)/16)*16;
    cout << "1. Computing Scores (Q * K^T)..." << endl;
    queue.enqueueNDRangeKernel(matmul_k, cl::NullRange, cl::NDRange(g_L,g_L), cl::NDRange(16,16));
    queue.finish(); // CRITICAL: Wait for MatMul to finish before Softmax starts

    // === STEP B: Softmax ===
    // Check your softmax.cl. If function name is "softmax_row", change "softmax" to "softmax_row"
    cl::Kernel softmax_k(progSoftMax, "softmax_row"); 
    softmax_k.setArg(0,d_Score);
    softmax_k.setArg(1,L);
    cout << "2. Applying Softmax..." << endl;
    queue.enqueueNDRangeKernel(softmax_k, cl::NullRange, cl::NDRange(L), cl::NullRange);
    queue.finish(); // CRITICAL: Wait for Softmax to finish before second MatMul

    // === STEP C: MatMul 2 (Scores * V) ===
    cl::Kernel matmul_v(progMatMul, "matmul");
    matmul_v.setArg(0,d_Score);
    matmul_v.setArg(1,d_V);
    matmul_v.setArg(2,d_Output);
    matmul_v.setArg(3,L);
    matmul_v.setArg(4,D);
    matmul_v.setArg(5,L);
    matmul_v.setArg(6,0); // No transpose

    int g_D = ((D + 15)/16)*16;
    cout << "3. Computing Output (Scores * V)..." << endl;
    queue.enqueueNDRangeKernel(matmul_v, cl::NullRange, cl::NDRange(g_D, g_L), cl::NDRange(16,16));
    queue.finish();

    // READ BACK
    queue.enqueueReadBuffer(d_Output, CL_TRUE, 0, sizeof(float)*Output.size(), Output.data());
    
    cout << "Attention Result[0]: " << Output[0] << endl;

    if (abs(Output[0] - 2.0f) < 0.1f) cout << "SUCCESS: Pipeline Works!" << endl;
    else {
        cout << "FAILURE: Math mismatch." << endl;
        // Debug: Read scores to see if softmax worked
        vector<float> debugScores(L*L);
        queue.enqueueReadBuffer(d_Score, CL_TRUE, 0, sizeof(float)*debugScores.size(), debugScores.data());
        cout << "Debug: Score[0] after Softmax: " << debugScores[0] << " (Expected 0.25)" << endl;
    }

    return 0;
}