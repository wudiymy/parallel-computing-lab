
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>


#define N   100000

__global__ void add(int *a, int *b, int *c) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += step;
	}
}

void cpu_add(int *a, int *b, int *c)
{
	for (int i = 0; i < N; i++)
		c[i] = a[i] + b[i];
}

int main(void) {
	int *a, *b, *c, *gpu_c;
	int *dev_a, *dev_b, *dev_c;

	a = (int*)malloc(N * sizeof(int));
	b = (int*)malloc(N * sizeof(int));
	c = (int*)malloc(N * sizeof(int));
	gpu_c = (int*)malloc(N * sizeof(int));

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	//init
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);
	int gpu_time, cpu_time;
	time_t begin, end;
	begin = clock();
	{
		add << <512, 512 >> > (dev_a, dev_b, dev_c);
		cudaThreadSynchronize();
	}
	end = clock();

	printf("GPU time = %d ms\n", gpu_time = int(end - begin));

	cudaMemcpy(gpu_c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	begin = clock();
	{
		cpu_add(a, b, c);
	}
	end = clock();

	printf("CPU time = %d ms\n", cpu_time = int(end - begin));

	bool success = true;
	for (int i = 0; i < N; i++) {
		if (c[i] != gpu_c[i]) {
			printf("Error:  %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}

	printf("speedup = %f\n", float(cpu_time) / gpu_time);

	if (success)
		printf("Everything is ok!\n");

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	free(a);
	free(b);
	free(c);

	return 0;
}


