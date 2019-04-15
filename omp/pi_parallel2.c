#include<stdio.h>
#include<omp.h>
#define STEPS 1000000000
#define NUM_THREADS 4

void main(){
    int i;
    double pi = 0, sum[NUM_THREADS];
    double step = 1.0 / (double) STEPS;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel 
    {
        double x;
        int id = omp_get_thread_num();
        sum[id] = 0;
        double sump = 0;
        #pragma omp for schedule(guided,7500)
        for(i=0; i < STEPS; i++){
            x = (i + 0.5) * step;
            sump += 4.0 / (1.0 + x*x);
        }
        sum[id] = sump;
    }

    for(i = 0, pi = 0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
	printf("\npi=%0.20lf\n", pi);
}