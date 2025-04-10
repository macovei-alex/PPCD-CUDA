#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <functional>
#include <iomanip>

__host__ int** alloc_matrix(int n, int m)
{
	int** mat = new int*[n];
	for (int i = 0;i < n;i++)
		mat[i] = new int[m];
	return mat;
}

__host__ void init_matrix(int**& mat, int n, int m)
{
	srand(time(NULL));
	mat = alloc_matrix(n, m);
	for (int i = 0;i < n;i++)
		for (int j = 0;j < m;j++)
			mat[i][j] = rand()%10;
}

__host__ void init_matrix_unit(int**& mat, int n, int m)
{
	mat = alloc_matrix(n, m);
	for (int i = 0;i < n;i++)
		for (int j = 0;j < m;j++)
			mat[i][j] = (i==j) ? 1 : 0;
}

__host__ void print_matrix(int**& mat, int n, int m)
{
	for (int i = 0;i < n;i++)
	{
		std::cout << std::endl;
		for (int j = 0;j < m;j++)
			std::cout << mat[i][j] << " ";
	}
	std::cout << std::endl << std::endl;
}

__host__ void free_matrix(int**& mat, int n)
{
	for (int i = 0;i < n;i++)
		delete mat[i];
	delete mat;
}

__host__ int** trans_matrix(int** a, int n, int m)
{
	int** b = alloc_matrix(m, n);
	for (int i = 0;i < n;i++)
		for (int j = 0;j < m;j++)
			b[j][i] = a[i][j];
	return b;
}

__host__ int** prod_matrix(int** a, int** b, int n, int m, int p)
{
	int** c = alloc_matrix(n, p);
	for (int i = 0;i < n;i++)
		for (int j = 0;j < p;j++)
		{
			c[i][j] = 0;
			for (int k = 0;k < m;k++)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	return c;
}

__host__ int** pseudo_prod_matrix(int** a, int** b, int n, int m, int p)
{
	int** c = alloc_matrix(n, p);
	for (int i = 0;i < n;i++)
		for (int j = 0;j < p;j++)
		{
			c[i][j] = 0;
			for (int k = 0;k < m;k++)
			{
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	return c;
}

__global__ void prod_matrix_on_device_1(int* a, int* b, int*&c, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = idx / n;
	int j = idx % n;
	if (i >= n || j >= n) return;
	int sum = 0;

	// a,b 1D array with n x n elements
	// c[i][j] = c[i*n+j]
	for (int k = 0;k < n;k++)
	{
		sum += a[i * n + k] * b[j * n + k];
	}
	c[i * n + j] = sum;
}

// Matrix 1D grid 2D
__global__ void prod_matrix_on_device_2(int* a, int* b, int*& c, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= n || j >= n) return;
	int sum = 0;

	for (int k = 0;k < n;k++)
	{
		sum += a[i * n + k] * b[j * n + k];
	}

	c[i * n + j] = sum;
}

__global__ void prod_matrix_on_device_3(int* A, int* B, int* C, int N) {
	const int TILE_SIZE = 16;
	__shared__ int shared_a[TILE_SIZE][TILE_SIZE];
	__shared__ int shared_b[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	for (int k = 0; k < N / TILE_SIZE; k++) {
		shared_a[threadIdx.y][threadIdx.x] = A[row * N + (k * TILE_SIZE + threadIdx.x)];
		shared_b[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * N + col];
		__syncthreads();
		for (int j = 0; j < TILE_SIZE; j++) {
			sum += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
		}
		__syncthreads();
	}
	C[row * N + col] = sum;
}

void deviceAllocMatrix(int*** d_mat, int N) {
	int** h_ptrs = (int**)malloc(N * sizeof(int*));
	int** d_ptrs;
	cudaMalloc(&d_ptrs, N * sizeof(int*));
	for (int i = 0; i < N; i++) {
		cudaMalloc(&h_ptrs[i], N * sizeof(int));
	}
	cudaMemcpy(d_ptrs, h_ptrs, N * sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat, &d_ptrs, sizeof(int**), cudaMemcpyHostToDevice);
	free(h_ptrs);
}

void deviceFreeMatrix(int** d_mat, int N) {
	int** h_ptrs = (int**)malloc(N * sizeof(int*));
	cudaMemcpy(h_ptrs, d_mat, N * sizeof(int*), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		cudaFree(h_ptrs[i]);
	}
	cudaFree(d_mat);
	free(h_ptrs);
}

__global__ void prod_matrix_on_device_4(int** A, int** B, int** C, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	for (int k = 0; k < N; k++) {
		sum += A[row][k] * B[k][col];
	}
	C[row][col] = sum;
}

int main()
{
	const int N = 1'000;
	int **a, **b, **c;
	init_matrix(a, N, N);
	init_matrix_unit(b, N, N);
	std::chrono::steady_clock::time_point start, end;
	std::chrono::duration<double> duration;

	const int TILE_SIZE = 16;
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 blocksPerGrid(N / TILE_SIZE, N / TILE_SIZE);

	int* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, N * N * sizeof(int));
	cudaMalloc((void**)&d_B, N * N * sizeof(int));
	cudaMalloc((void**)&d_C, N * N * sizeof(int));

	int** d_A2D;
	int** d_B2D;
	int** d_C2D;
	deviceAllocMatrix(&d_A2D, N);
	deviceAllocMatrix(&d_B2D, N);

	start = std::chrono::high_resolution_clock::now();
	deviceAllocMatrix(&d_C2D, N);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds - ALLOC\n";


	// prod_matrix
	start = std::chrono::high_resolution_clock::now();
	c = prod_matrix(a, b, N, N, N);
	end = std::chrono::high_resolution_clock::now();	
	duration = end - start;
	std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds - prod_matrix\n";

	// pseudo_prod_matrix
	start = std::chrono::high_resolution_clock::now();
	c = pseudo_prod_matrix(a, trans_matrix(b, N, N), N, N, N);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds - pseudo_prod_matrix\n";

	// V1
	start = std::chrono::high_resolution_clock::now();
	int blockSize = 256;
	int gridSize = (N * N + blockSize - 1) / blockSize;
	prod_matrix_on_device_1 << <gridSize, blockSize >> > (*a, *b, *c, N);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds - V1\n";

	// V2
	start = std::chrono::high_resolution_clock::now();
	prod_matrix_on_device_2 << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds - V2\n";

	// V3
	start = std::chrono::high_resolution_clock::now();
	prod_matrix_on_device_3 << <blocksPerGrid, threadsPerBlock>> > (d_A, d_B, d_C, N);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds - V3\n";

	// V4
	start = std::chrono::high_resolution_clock::now();
	prod_matrix_on_device_4 << <blocksPerGrid, threadsPerBlock >> > (d_A2D, d_B2D, d_C2D, N);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	std::cout << "Execution time: " << std::fixed << std::setprecision(9) << duration.count() << " seconds - V4\n";

	free_matrix(a, N);
	free_matrix(b, N);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	deviceFreeMatrix(d_A2D, N);
	deviceFreeMatrix(d_B2D, N);
	deviceFreeMatrix(d_C2D, N);

	return 0;
}