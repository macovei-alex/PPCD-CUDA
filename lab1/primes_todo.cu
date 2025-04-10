#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

__global__ void helloCUDA() {
    printf("Hello from CUDA Kernel!\n");
}

bool is_prime_host(int num) {
    // Check if a number is prime or not
    return true;
}

__host__ int count_primes_onHost(int N) {
    // Count all prime numbers from [1,N] interval
    return 0;
}

__device__ bool is_prime_device(int num) {
    // The same logic as is_prime_host() but marked as __device__
    return true;
}

__global__ void count_primes_device(int *d_count, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Count all prime numbers from [index,N] interval
    // Use atomicAdd for increase d_count value
}

// Wrapper function to launch the CUDA kernel
int count_primes_onDevice(int N) {
    int *d_count;
    int h_count = 0;

    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    // Define threadsPerBlock and blocksPerGrid
    // Launch device function based on these values
    
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    
    return h_count;
}

// CUDA kernel using shared memory to count primes
__global__ void count_primes_shared(int *d_count, int N) {
    // Use a shared memory array for block results

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int thread_id = threadIdx.x;

    // Initialize shared memory

    __syncthreads();

    // Each thread processes multiple numbers
   
    __syncthreads();

    // Reduce within the block using the first thread
    if (thread_id == 0) {
        int block_sum = 0;
        // Use atomicAdd for increase d_count value
    }
}

int count_primes_onShared(int N) {
    int *d_count;
    int h_count = 0;

    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_primes_shared<<<blocksPerGrid, threadsPerBlock>>>(d_count, N);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    return h_count;
}

int main() {
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();

    const int N = 10'000'000;
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> duration;
    int count = 0;

    start = std::chrono::high_resolution_clock::now();
    // count = count_primes_onHost(N);
    // std::cout << "Count: " << count << " prime numbers\n";
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    // std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds\n\n";

    // Repeat for count_primes_onDevice and count_primes_onShared


    return 0;
}
