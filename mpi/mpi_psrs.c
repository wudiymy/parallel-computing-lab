#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

int DATA_NUM=10000;

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

    int datas_begin[mpi_size];
    int datas_len[mpi_size];
    if(my_rank == 0)
    {
        datas_init(datas);
        for(int i = 0; i < mpi_size - 1; i++)
        {
            datas_begin[i] = i * length;
            datas_len[i] = length;
        }
        datas_begin[mpi_size - 1] = (mpi_size - 1) * length;
        datas_len[mpi_size - 1] = DATA_NUM - (mpi_size - 1) * length;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //step1: evenly divided 
    MPI_Scatterv(datas, datas_len, datas_begin, MPI_INT, local_datas, 
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
                mpi_size, MPI_INT, 0, MPI_COMM_WORLD);

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
    int data_index = 0;
    for(int i = 0; i < mpi_size - 1; i++)
    {
        part_start_index[i] = data_index;
        part_length[i] = 0;

        while((data_index < local_datas_size) && (local_datas[data_index] <= privots[i]))
        {
            data_index++;
            part_length[i]++;
        }
    }
    part_start_index[mpi_size - 1] = data_index;
    part_length[mpi_size - 1] = local_datas_size - data_index;

    MPI_Barrier(MPI_COMM_WORLD);

    //step7: global exchange

    int recv_rank_partlen[mpi_size];
    MPI_Alltoall(part_length, 1, MPI_INT, recv_rank_partlen, 1, MPI_INT, MPI_COMM_WORLD);

    int rank_partlen_sum = 0;
    int rank_part_start[mpi_size];

    for(int i = 0; i < mpi_size; i++)
    {
        rank_part_start[i] = rank_partlen_sum;
        rank_partlen_sum += recv_rank_partlen[i];
    }

    int recv_part_data[rank_partlen_sum];

    MPI_Alltoallv(local_datas, part_length, part_start_index, MPI_INT, 
                    recv_part_data, recv_rank_partlen, rank_part_start, MPI_INT, MPI_COMM_WORLD);

    qsort(recv_part_data, rank_partlen_sum, sizeof(int), cmp);

    MPI_Barrier(MPI_COMM_WORLD);

    //step8

    int master_partlen[mpi_size];
    int master_data_start[mpi_size];

    MPI_Gather(&rank_partlen_sum, 1, MPI_INT, master_partlen, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(my_rank == 0)
    {
        for(int i = 0, temp_sum = 0; i < mpi_size; i++)
            {
                master_data_start[i] = temp_sum;
                temp_sum += master_partlen[i]; 
            }
    }

    MPI_Gatherv(recv_part_data, rank_partlen_sum, MPI_INT, 
                    datas, master_partlen, master_data_start, MPI_INT, 0, MPI_COMM_WORLD);

    free(local_datas);
    if(my_rank == 0)
    {
        if(datas_check(datas))
            printf("\nyou are right\n");
        else
            printf("\nsomething wrong\n");
    }

    MPI_Finalize();
    return 0;
}