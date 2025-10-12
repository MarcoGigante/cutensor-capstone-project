/* 
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   - Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   - Neither the names of copyright holders nor the names of its contributors
 *     may be used to endorse or promote products derived from this software 
 *     without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * ---------------------------------------------------------------------------
 * Matrix Multiplication via cuTENSOR: GPU Tensor Contraction Example
 * ---------------------------------------------------------------------------
 *
 * This program demonstrates the application of high-performance tensor
 * contraction for matrix multiplication on NVIDIA GPUs via the cuTENSOR library.
 * The implementation initializes random input matrices, performs the contraction
 * corresponding to C = alpha * A * B + beta * C using cuTENSOR's optimized APIs,
 * and outputs the results.
 * 
 * The code provides a reproducible computational workflow suitable for research,
 * benchmarking, or advanced educational purposes. 
 */


#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <unordered_map>
#include <vector>

// Error handling macros
#define HANDLE_ERROR(x) { \
  cutensorStatus_t err = x; \
  if (err != CUTENSOR_STATUS_SUCCESS) { \
    printf("cuTENSOR error: %s\n", cutensorGetErrorString(err)); exit(-1); \
  } \
}

#define HANDLE_CUDA_ERROR(x) { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(-1); \
    } \
}

int main()
{
    typedef float floatType;
    // Matrix sizes: C[m, k] = sum_h A[m, h] * B[h, k]
    const int M = 96, H = 64, K = 128;

    // Mode labels ('m', 'h', 'k')
    std::vector<int> modeA{'m','h'};
    std::vector<int> modeB{'h','k'};
    std::vector<int> modeC{'m','k'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // Extents mapping ('m','h','k')
    std::unordered_map<int, int64_t> extent;
    extent['m'] = M;
    extent['h'] = H;
    extent['k'] = K;

    // Create vectors of extents for each tensor
    std::vector<int64_t> extentA, extentB, extentC;
    for(auto mode : modeA) extentA.push_back(extent[mode]);
    for(auto mode : modeB) extentB.push_back(extent[mode]);
    for(auto mode : modeC) extentC.push_back(extent[mode]);

    // Number of elements and allocation
    size_t elementsA = 1, elementsB = 1, elementsC = 1;
    for(auto mode : modeA) elementsA *= extent[mode];
    for(auto mode : modeB) elementsB *= extent[mode];
    for(auto mode : modeC) elementsC *= extent[mode];
    size_t sizeA = sizeof(floatType) * elementsA;
    size_t sizeB = sizeof(floatType) * elementsB;
    size_t sizeC = sizeof(floatType) * elementsC;

    // Allocate on host
    floatType *A = (floatType*) malloc(sizeA);
    floatType *B = (floatType*) malloc(sizeB);
    floatType *C = (floatType*) malloc(sizeC);
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5f)*10;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5f)*10;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = 0.0f;

    // Allocate on device
    void *A_d, *B_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc(&A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc(&B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc(&C_d, sizeC));

    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    // Alignment for cuTENSOR (128 bytes)
    const uint32_t kAlignment = 128;
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    // Initialize cuTENSOR handle
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t descA, descB, descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, nmodeA, extentA.data(), NULL, CUTENSOR_R_32F, kAlignment));
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descB, nmodeB, extentB.data(), NULL, CUTENSOR_R_32F, kAlignment));
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, nmodeC, extentC.data(), NULL, CUTENSOR_R_32F, kAlignment));

    // Create Contraction Descriptor
    cutensorOperationDescriptor_t contractionDesc;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
    HANDLE_ERROR(cutensorCreateContraction(handle,
        &contractionDesc,
        descA, modeA.data(), CUTENSOR_OP_IDENTITY,
        descB, modeB.data(), CUTENSOR_OP_IDENTITY,
        descC, modeC.data(), CUTENSOR_OP_IDENTITY,
        descC, modeC.data(),
        descCompute));

    // Algorithm & Plan Preferences
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE));

    // Workspace Estimation
    uint64_t workspaceSizeEstimate = 0, actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, contractionDesc, planPref, CUTENSOR_WORKSPACE_DEFAULT, &workspaceSizeEstimate));

    // Create Plan
    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle, &plan, contractionDesc, planPref, workspaceSizeEstimate));

    // Query actual workspace
    HANDLE_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize, sizeof(actualWorkspaceSize)));

    void *workspace = nullptr;
    if (actualWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&workspace, actualWorkspaceSize));
        assert(uintptr_t(workspace) % 128 == 0);
    }

    // CUDA stream
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    // Do tensor contraction: C = alpha * A * B + beta * C
    float alpha = 1.0f, beta = 0.0f;
    HANDLE_ERROR(cutensorContract(handle, plan,
        (void*) &alpha, A_d, B_d,
        (void*) &beta, C_d, C_d,
        workspace, actualWorkspaceSize, stream));

    // Copy result to host
    HANDLE_CUDA_ERROR(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Save results to a text file
    std::ofstream fout("results/matrix_output.txt");
    if(fout.is_open()) {
        for(int m=0; m<M; ++m){
            for(int k=0; k<K; ++k) {
                fout << C[m*K + k] << " ";
            }
            fout << "\n";
        }
        fout.close();
        printf("Results written to results/matrix_output.txt\n");
    }
    else {
        printf("Unable to open results/matrix_output.txt for writing!\n");
    }

    // Cleanup
    cutensorDestroyPlan(plan);
    cutensorDestroyOperationDescriptor(contractionDesc);
    cutensorDestroyTensorDescriptor(descA);
    cutensorDestroyTensorDescriptor(descB);
    cutensorDestroyTensorDescriptor(descC);
    cutensorDestroy(handle);
    cudaStreamDestroy(stream);
    if(workspace) cudaFree(workspace);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    free(A); free(B); free(C);
    return 0;
}

