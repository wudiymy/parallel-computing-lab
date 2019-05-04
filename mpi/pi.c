#include<stdio.h>
#include<math.h>
#include<mpi.h>

#define STEPS 1000000000
int min(int a, int b)
{
    int result;
    if(a < b)
        result = a;
    else 
        result = b;

    return result;
}
int main( int argc, char * argv[])
{
    int my_rank;
    int mpi_size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    int length = (STEPS + mpi_size - 1) / mpi_size;
    double step = 1.0 / STEPS;
    double x, local_pi = 0;
    double global_pi;
    int my_begin = length * my_rank;
    int my_end = min(length * (my_rank + 1), STEPS);

    for(int i = my_begin; i < my_end; i++)
    {
        x = ( i + 0.5 ) * step;
        local_pi += 4.0 / (1.0 + x * x);
    }

    local_pi *= step;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(my_rank == 0)
        printf("\n pi = %0.15lf\n",global_pi);

    MPI_Finalize();

    return 0;
    
}