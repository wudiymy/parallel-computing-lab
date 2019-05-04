#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define DATA_NUM 10000

void datas_init(int *datas)
{
    for(int i = 0; i < DATA_NUM; i++)
        datas[i] = rand();
}

int datas_check(const int *datas)
{
    int i;
    for(int i = 0; i < DATA_NUM - 1; i++)
    {
        if( datas[i] > datas[i+1] )
            break;
    }
    return (i == DATA_NUM - 1) ? 1 : 0;
}

int cmp(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}

int min(const int a, const int b)
{
    return ( a > b )? b : a;
}

int main()
{
    int mpi_size;
    int my_rank;
    int datas[DATA_NUM];

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int length = (DATA_NUM + mpi_size - 1) / mpi_size;
    int offset = length * my_rank;
    int local_datas_size = min(length, DATA_NUM - length * my_rank);
    int *local_datas = (int *) malloc (sizeof(int) * local_datas_size);

    if(my_rank == 0)
    {
        datas_init(datas);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //step1: evenly divided 
    MPI_Scatterv(datas, DATA_NUM, offset, MPI_INT, local_datas, 
                    local_datas_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    //step2: local sort
    qsort(local_datas, local_datas_size, sizeof(int), cmp);
    
    //step3: select samples
    int regular_samples[mpi_size];
    for(int i = 0; i < mpi_size; i++)
        regular_samples[i] = local_datas[i * local_datas_size / mpi_size];

    MPI_Barrier(MPI_COMM_WORLD);

    int global_regular_samples[mpi_size * mpi_size];

    MPI_Gather(regular_samples, mpi_size, MPI_INT, global_regular_samples, 
                mpi_size * mpi_size, MPI_INT, 0, MPI_COMM_WORLD);

    int privots[mpi_size - 1];

    if(my_rank == 0)
    {
        qsort(global_regular_samples, mpi_size * mpi_size, sizeof(int), cmp);
        for(int i = 0; i < mpi_size - 1; i++)
        {
            privots[i] = global_regular_samples[(i+1) * mpi_size];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(privots, mpi_size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    int part_start_index[mpi_size];
    int part_length[mpi_size];

    for(int i = 0, data_index = 0; i < mpi_size - 1; i++)
    {
        
    }


}