#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

using namespace std;

// Helper to check errors immediately
void checkErr(cl_int err, const char* name) {
    if (err != CL_SUCCESS) {
        cerr << "ERROR: " << name << " (" << err << ")" << endl;
        exit(1);
    }
}

int main() {
    cl_int err; // Store error codes here

    // 1. SETUP
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) { cerr << "No platforms found" << endl; return 1; }
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl;

    // 2. READ KERNEL FILE (CRITICAL CHECK)
    // NOTE: Check this path! If you are in 'Phase_2' folder, correct path is likely "kernels/mm_tiled.cl"
    string path = "kernels/mm_tiled.cl"; 
    ifstream file(path);
    if (!file.is_open()) { 
        // Try the other common path just in case
        path = "Phase2_Math/kernels/mm_tiled.cl";
        file.open(path);
        if(!file.is_open()) {
            cerr << "CRITICAL ERROR: Could not open kernel file at 'kernels/mm_tiled.cl' OR 'Phase2_Math/kernels/mm_tiled.cl'" << endl;
            cerr << "Current file path check failed." << endl;
            return 1;
        }
    }
    cout << "Loaded kernel from: " << path << endl;
    
    string src((istreambuf_iterator<char>(file)), (istreambuf_iterator<char>()));
    if(src.empty()) { cerr << "ERROR: Kernel source file is empty!" << endl; return 1; }

    cl::Program program(context, src);
    if(program.build({device}) != CL_SUCCESS) {
        cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return 1;
    }

    // 3. DIMENSIONS
    cl_int M = 512, N = 512, K = 512;
    vector<float> A(M*K, 1.0f);
    vector<float> B(K*N, 2.0f);
    vector<float> C(M*N, 0.0f);

    // 4. BUFFERS (With explicit error checking)
    cout << "Creating Buffers..." << endl;
    cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*A.size(), A.data(), &err);
    checkErr(err, "Buffer A Creation");
    
    cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*B.size(), B.data(), &err);
    checkErr(err, "Buffer B Creation");
    
    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float)*C.size(), nullptr, &err);
    checkErr(err, "Buffer C Creation");

    // 5. SET ARGUMENTS (Check each one!)
    cl::Kernel kernel(program, "mm_tiled", &err);
    checkErr(err, "Kernel Creation 'mm_tiled'");

    cout << "Setting Arguments..." << endl;
    checkErr(kernel.setArg(0, bufA), "Arg 0 (bufA)");
    checkErr(kernel.setArg(1, bufB), "Arg 1 (bufB)");
    checkErr(kernel.setArg(2, bufC), "Arg 2 (bufC)");
    checkErr(kernel.setArg(3, sizeof(cl_int), &M), "Arg 3 (M)"); // Explicit size/pointer syntax
    checkErr(kernel.setArg(4, sizeof(cl_int), &N), "Arg 4 (N)");
    checkErr(kernel.setArg(5, sizeof(cl_int), &K), "Arg 5 (K)");

    // 6. EXECUTION
    cl::NDRange globalSize(N, M);   
    cl::NDRange localSize(16, 16); 

    cout << "Launching Kernel..." << endl;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    checkErr(err, "EnqueueNDRangeKernel");
    
    queue.finish();
    
    // 7. READ BACK
    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float)*C.size(), C.data());
    cout << "Success! C[0] = " << C[0] << endl;

    return 0;
}