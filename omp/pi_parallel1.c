#include<stdio.h>
#include<omp.h>
#define STEPS 1000000000
#define NUM_THREADS 8

void main(){
    int i;
	double x, pi, sum[NUM_THREADS];
	double step = 1.0 / STEPS;
    omp_set_num_threads(NUM_THREADS);
    
    #pragma omp parallel private(i,x)
    {
        double sum_part = 0;
        int id = omp_get_thread_num();
	    for(i = id; i < STEPS; i+=NUM_THREADS){
	    	x = (i + 0.5) * step;
	    	sum_part += 4.0/(1.0 + x * x);
	    }
        sum[id] = sum_part;
    }
	
    pi = 0;
    for(i = 0; i < NUM_THREADS; i++)
        pi += sum[i];
    pi *= step; 
	printf("\npi=%0.20lf\n", pi);
}