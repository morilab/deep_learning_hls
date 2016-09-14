#include "test.h"


int main(void) {
	Matrix<1,3,int> z;

	//------------------------------------
	// setting input data (x)
	//------------------------------------
	int x[4] =
		{1,3,5,7};

	//------------------------------------
	// setting Parameter (bias,weight)
	//------------------------------------
	int bias[3] =
		{-1,0,1};

	int weight[3][4] = {
		{ 2, 2, 2, 2},
		{ 2, 2,-8,-4},
		{ 2,-1,-2, 2},
	};

	//------------------------------------
	// View Input data & Parameter
	//------------------------------------
	for(int i=0;i<4;i++){
		printf("x%d = %d\n",i,x[i]);
	}
	printf("\n");

	printf("Weight = {\n");
	for(int i=0;i<4;i++){
		printf("  {");
		for(int j=0;j<3;j++){
			printf(" %3d",weight[j][i]);
		}
		printf(" }\n");
	}
	printf("};\n\n");

	printf("Bias = {\n  {");
	for(int i=0;i<3;i++){
		printf(" %3d",bias[i]);
	}
	printf(" }\n};\n\n");

	//------------------------------------
	// run DeepLearning
	//------------------------------------
	z = deep_learning<4,3,int>(x, weight, bias);

	for(int i=0;i<3;i++){
		printf("u%d = %d\n",i,z(0,i));
	}



	//------------------------------------
	convolution_perceptron<28,28,5,5,int> tr;
	Matrix<28,28,int> inframe;

	for(int y=0;y<28;y++){
		for(int x=0;x<28;x++){
			inframe(x,y)= 3;
		}
	}
	int filter[] = {
		0,1,1,1,2,
		0,2,5,2,1,
		0,2,5,2,1,
		0,2,5,5,1,
		1,0,0,0,0,
	};
	tr.set_filter(filter);
	tr.in = inframe;
	tr.Proc();
	tr.view_filter();
	tr.view_conv();
	tr.view_out();

	return 0;
}

