#include<stdio.h>
#include<omp.h>
#define STEPS 1000000000
#define NUM_THREADS 4

void main()
{
    double pi = 0, sum = 0, x = 0;
    double step = 1.0 / (double) STEPS;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel private(x, sum)
    {
        int i;
        int id = omp_get_thread_num();
        for(i=id, sum=0; i < STEPS; i+=NUM_THREADS)
        {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x*x);
        }

        #pragma omp critical
            pi += sum;
    }
    pi *= step;
    printf("\npi=%0.20lf\n", pi);
}