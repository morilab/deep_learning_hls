#include "test.h"

int main(void) {
	Matrix<1,4,int> x;
	Matrix<1,3,int> z;
	Matrix<4,3,int> weight;
	Matrix<1,3,int> bias;

	x(0,0) = 1;
	x(0,1) = 2;
	x(0,2) = 3;
	x(0,3) = 4;

	for(int j=0;j<3;j++){
		bias(0,j)=j-1;
		for(int i=0;i<4;i++){
			weight(i,j)=2;
		}
	}

	for(int i=0;i<4;i++){
		printf("x%d = %d\n",i,x(0,i));
	}
	printf("\n");

	printf("Weight = {\n");
	for(int j=0;j<3;j++){
		printf("  {");
		for(int i=0;i<4;i++){
			printf(" %3d",weight(i,j));
		}
		printf(" }\n");
	}
	printf("};\n");

	printf("Bias = {\n");
	printf("  {");
	for(int i=0;i<3;i++){
		printf(" %3d",bias(0,i));
	}
	printf(" }\n");
	printf("};\n");

	z = deep_learning<4,3,int>(x, weight, bias);

	for(int i=0;i<3;i++){
		printf("u%d = %d\n",i,z(0,i));
	}

	return 0;
}
