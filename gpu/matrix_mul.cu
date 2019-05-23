
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>


#define BLOCK_SIZE 16
#define BLOCK_X 100
#define BLOCK_Y 100
#define IMP 100

//BLOCK_X should be smaller than IMP

#define WA (BLOCK_X * BLOCK_SIZE) // Matrix A width
#define HA (BLOCK_Y * BLOCK_SIZE) // Matrix A height
#define WB (IMP * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

__device__  float * GetSubMatrix(float *matrix, int m, int index, int width)
{
	return  matrix + width * BLOCK_SIZE*index + BLOCK_SIZE * m;
}

__global__ void matrix_mul_gpu(float *C, float * A, float *B, int wA, int hA, int wB)
{
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Csub = 0;
	int end = wA / BLOCK_SIZE;
	for (int m = 0; m < end; m++)
	{
		float *subA = GetSubMatrix(A, m, by, wA);
		float *subB = GetSubMatrix(B, bx, m, wB);
		As[ty][tx] = *(subA + wA * ty + tx);
		Bs[ty][tx] = *(subB + wB * ty + tx);

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];

		__syncthreads();
	}

	float *subC = GetSubMatrix(C, bx, by, wB);
	*(subC + wB * ty + tx) = Csub;
}

void matrix_mul_cpu(float *C, float *A, float *B)
{

	for (int i = 0; i < HC; i++)
		for (int j = 0; j < WC; j++)
		{
			float sum = 0;
			for (int k = 0; k < WA; k++)
				sum += A[i * WA + k] * B[k * WB + j];
			C[i * WC + j] = sum;
		}
}

void matrix_init(float *matrix, int size)
{
	for (int i = 0; i < size; i++)
		matrix[i] = rand() / (float)RAND_MAX;
}

void matrix_print(float *matrix, int H, int W)
{
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			printf("%f ", matrix[W * i + j]);
		}
		printf("\n");
	}
}
bool matrix_check(float *matrix1, float *matrix2, int size)
{
	int result = true;
	for (int i = 0; i < size; i++)
	{
		if (fabsf(matrix1[i] - matrix2[i]) > 1e-3)
		{
			printf("error: %d %lf != %lf\n", i, matrix1[i], matrix2[i]);
			result = false;
		}
	}
	return result;
}

int main()
{
	float *A, *B, *C, *C_gpu;
	float *dev_A, *dev_B, *dev_C;
	int size_A, size_B, size_C;
	size_A = WA * HA;
	size_B = WB * HB;
	size_C = WC * HC;
	A = (float *)malloc(size_A * sizeof(float));
	B = (float *)malloc(size_B * sizeof(float));
	C = (float *)malloc(size_C * sizeof(float));
	C_gpu = (float *)malloc(size_C * sizeof(float));
	cudaMalloc((void **)&dev_A, size_A * sizeof(float));
	cudaMalloc((void **)&dev_B, size_B * sizeof(float));
	cudaMalloc((void **)&dev_C, size_C * sizeof(float));

	matrix_init(A, size_A);
	matrix_init(B, size_B);

	cudaMemcpy(dev_A, A, size_A * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, size_B * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(WC / threads.x, HC / threads.y);
	//printf("A\n");
	//matrix_print(A, HA, WA);
	//printf("B\n");
	//matrix_print(B, HB, WB);
	//printf("\n");

	clock_t start, stop;
	int cpu_time;
	int gpu_time;
	start = clock();
	{
		matrix_mul_gpu << <grid, threads >> > (dev_C, dev_A, dev_B, WA, HA, WB);
		cudaThreadSynchronize();
	}
	stop = clock();
	printf("gpu_time = %d ms\n", gpu_time = (stop - start));

	clock_t begin, end;
	begin = clock();
	{
		matrix_mul_cpu(C, A, B);
	}
	end = clock();
	printf("cpu_time = %d ms\n", cpu_time = (end - begin));

	printf("SpeedUp = %f\n", double(cpu_time) / (gpu_time));
	

	cudaMemcpy(C_gpu, dev_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

	if (matrix_check(C, C_gpu, size_C))
		printf("Everything is ok\n");

	//printf("C\n");
	//matrix_print(C, HC, WC);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	
	free(A);
	free(B);
	free(C);
	free(C_gpu);
	return 0;
}
