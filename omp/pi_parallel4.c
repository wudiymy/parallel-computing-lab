#include<stdio.h>
#include<omp.h>
#define STEPS 1000000000
#define NUM_THREADS 4

void main()
{
    double pi, sum = 0, x;
    double step = 1.0 / (double) STEPS;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel for reduction(+:sum) private(x)
    for(int i=0; i < STEPS; i++) 
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x*x);
    }

    pi = sum * step;
    printf("\npi=%0.20lf\n", pi);
}