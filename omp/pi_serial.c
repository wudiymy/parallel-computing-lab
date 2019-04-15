#include<stdio.h>

#define steps 1000000000

void main(){
	double x, pi, sum = 0;
	double step = 1.0 / steps;
	for(int i = 1; i < steps; i++){
		x = (i - 0.5) * step;
		sum = sum + 4.0/(1.0 + x * x);
	}
	pi = step * sum;
	printf("\npi=%0.20lf\n", pi);
}

