#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#define NUM_THREADS 8
#define NUM_DATA 100000



void datas_init(int *datas)
{
	for(int i = 0; i < NUM_DATA; i++)
		datas[i] = rand();

}

int datas_check(int *datas)
{
    int i;
	for(i = 0; i < NUM_DATA-1; i++)
	{        
		if( datas[i] > datas[i+1] )
			break;
	}

	return (i == NUM_DATA - 1)?1:0;
}

void datas_sort(int *datas, int length)
{

    int *p = datas;
	for(int i = 0; i < length; i++)
		for(int j = i + 1; j < length; j++)
		{
			if(p[i] > p[j])
			{
				p[i] = p[i] ^ p[j];
				p[j] = p[i] ^ p[j];
				p[i] = p[i] ^ p[j];
			}
		}	
}

int main()
{
    //int datas[NUM_DATA] = {48,39,6,72,91,14,69,40,89,61,12,21,84,58,32,33,72,20};
	int length = NUM_DATA / NUM_THREADS;

	if(length == 0)
		return 0;
	int datas[NUM_DATA];
	datas_init(datas);



	int regularSamples[NUM_THREADS * NUM_THREADS]; //样本数组
	int privots[NUM_THREADS - 1];  //选取的主元数组
	
	int partStartIndex[NUM_THREADS * NUM_THREADS]; // 主元划分标志位
	int partLength[NUM_THREADS * NUM_THREADS];  //划分长度
	int tt = 0;

	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel shared(regularSamples, privots) 
	{
		int id = omp_get_thread_num();
		int idStart = id * length;
		int *thread_datas = datas + idStart;


		datas_sort(thread_datas, length);


#pragma omp critical  //bug
		for(int i = 0; i < NUM_THREADS; i++)
		{
			regularSamples[NUM_THREADS * id + i] = thread_datas[(i * length) / NUM_THREADS];
		}
#pragma omp barrier
#pragma omp single 
		{

			datas_sort(regularSamples, NUM_THREADS * NUM_THREADS);
			for(int i = 0; i < NUM_THREADS - 1; i++)  //选取p-1个主元
				privots[i] = regularSamples[(i+1) * NUM_THREADS];

		}
#pragma omp barrier

		

		int dataIndex = 0;
        int anotherIdStart = id * NUM_THREADS;
		for(int i = 0; i < NUM_THREADS - 1; i++)
		{
			partStartIndex[i + anotherIdStart] = dataIndex;
			partLength[i + anotherIdStart] = 0;
			
			while((dataIndex < length) && (thread_datas[dataIndex] <= privots[i]))
			{
				dataIndex++;
				(partLength[i + anotherIdStart])++;
			}
		}
		
		partStartIndex[NUM_THREADS-1 + anotherIdStart] = dataIndex;
		partLength[NUM_THREADS-1 + anotherIdStart] = length - dataIndex;

#pragma omp barrier
		//全局交换
		int size = 0;
		for(int i = 0; i < NUM_THREADS; i++)
			size += partLength[id + i * NUM_THREADS];


#pragma omp barrier
		int temp[size];
		int index;
		int len;
		for(int i = 0,k=0; i < NUM_THREADS; i++)
		{
			index = partStartIndex[id + i * NUM_THREADS] + i * length;
			len = partLength[id + i * NUM_THREADS];
			for(int j = 0; j < len; j++)
			{
				temp[k++] = datas[index + j];
			}
		}
		
		datas_sort(temp, size);
		  
        
#pragma omp for ordered schedule(static,1)
    for( int t=0; t<omp_get_num_threads(); ++t )
    {
#pragma omp ordered
        {
            for(int i = 0; i < size; i++)
				datas[tt++] = temp[i];
        }
    }        
		

	}
	
	
	if(datas_check(datas))
		printf("YOU ARE RIGHT\n");
	else
		printf("SOMETHINE WRONG\n");
	
	return 0;
}
	
	
	
	
	
	